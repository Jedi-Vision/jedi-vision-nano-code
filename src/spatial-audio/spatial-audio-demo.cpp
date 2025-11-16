#include <phonon.h>
#include <fmt/format.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

// WAV file header structures
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize;
};

struct WAVFile {
    uint32_t sampleRate;
    uint16_t numChannels;
    uint16_t bitsPerSample;
    std::vector<float> samples; // Mono samples as 32-bit float
};

// Read WAV file
bool readWAV(const std::string& filename, WAVFile& wav) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    // Validate RIFF header
    if (std::memcmp(header.riff, "RIFF", 4) != 0 || std::memcmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Error: Not a valid WAV file" << std::endl;
        return false;
    }

    // Handle chunk-based file structure
    if (std::memcmp(header.fmt, "fmt ", 4) != 0) {
        // Skip until we find fmt chunk
        file.seekg(12);
        char chunkID[4];
        uint32_t chunkSize;
        while (file.read(chunkID, 4) && file.read(reinterpret_cast<char*>(&chunkSize), 4)) {
            if (std::memcmp(chunkID, "fmt ", 4) == 0) {
                file.read(reinterpret_cast<char*>(&header.audioFormat), 2);
                file.read(reinterpret_cast<char*>(&header.numChannels), 2);
                file.read(reinterpret_cast<char*>(&header.sampleRate), 4);
                file.read(reinterpret_cast<char*>(&header.byteRate), 4);
                file.read(reinterpret_cast<char*>(&header.blockAlign), 2);
                file.read(reinterpret_cast<char*>(&header.bitsPerSample), 2);
                if (chunkSize > 16) {
                    file.seekg(chunkSize - 16, std::ios::cur);
                }
                break;
            } else {
                file.seekg(chunkSize, std::ios::cur);
            }
        }
        
        // Find data chunk
        file.read(chunkID, 4);
        file.read(reinterpret_cast<char*>(&header.dataSize), 4);
        while (std::memcmp(chunkID, "data", 4) != 0) {
            file.seekg(header.dataSize, std::ios::cur);
            file.read(chunkID, 4);
            file.read(reinterpret_cast<char*>(&header.dataSize), 4);
        }
    }

    wav.sampleRate = header.sampleRate;
    wav.numChannels = header.numChannels;
    wav.bitsPerSample = header.bitsPerSample;

    if (header.audioFormat != 1) {
        std::cerr << "Error: Only PCM format is supported" << std::endl;
        return false;
    }

    // Read audio data
    size_t numSamples = header.dataSize / (header.bitsPerSample / 8) / header.numChannels;
    std::vector<int16_t> rawSamples(numSamples * header.numChannels);
    file.read(reinterpret_cast<char*>(rawSamples.data()), header.dataSize);

    // Convert to mono float32
    wav.samples.resize(numSamples);
    const float scale = 1.0f / 32768.0f;
    
    if (header.numChannels == 1) {
        for (size_t i = 0; i < numSamples; i++) {
            wav.samples[i] = rawSamples[i] * scale;
        }
    } else {
        // Downmix stereo to mono
        for (size_t i = 0; i < numSamples; i++) {
            float left = rawSamples[i * header.numChannels] * scale;
            float right = rawSamples[i * header.numChannels + 1] * scale;
            wav.samples[i] = (left + right) * 0.5f;
        }
    }

    file.close();
    std::cout << fmt::format("Loaded WAV: {} Hz, {} channels, {} samples\n", 
                             wav.sampleRate, header.numChannels, numSamples);
    return true;
}

// Write WAV file
bool writeWAV(const std::string& filename, const std::vector<float>& leftChannel, 
              const std::vector<float>& rightChannel, uint32_t sampleRate) {
    if (leftChannel.size() != rightChannel.size()) {
        std::cerr << "Error: Channel size mismatch" << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file: " << filename << std::endl;
        return false;
    }

    WAVHeader header;
    header.numChannels = 2;
    header.sampleRate = sampleRate;
    header.bitsPerSample = 16;
    header.byteRate = sampleRate * header.numChannels * header.bitsPerSample / 8;
    header.blockAlign = header.numChannels * header.bitsPerSample / 8;
    header.dataSize = leftChannel.size() * header.numChannels * header.bitsPerSample / 8;
    header.chunkSize = 36 + header.dataSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // Write interleaved stereo data as 16-bit PCM
    const float scale = 32767.0f;
    for (size_t i = 0; i < leftChannel.size(); i++) {
        int16_t left = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, leftChannel[i])) * scale);
        int16_t right = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, rightChannel[i])) * scale);
        file.write(reinterpret_cast<const char*>(&left), sizeof(int16_t));
        file.write(reinterpret_cast<const char*>(&right), sizeof(int16_t));
    }

    file.close();
    std::cout << fmt::format("Wrote WAV: {} Hz, stereo, {} samples\n", sampleRate, leftChannel.size());
    return true;
}

// Generate smooth 3D trajectory around listener
// Returns both the direction vector and the actual distance
struct TrajectoryResult {
    IPLVector3 direction;
    float distance;
};

TrajectoryResult generateTrajectory(float t, float duration) {
    // Normalize time to [0, 1]
    float normalizedTime = t / duration;
    
    // Create a fast, dynamic trajectory that covers all directions
    // Multiple cycles for faster movement
    float speedMultiplier = 3.0f; // Makes movement 3x faster
    float timeScaled = std::fmod(normalizedTime * speedMultiplier, 1.0f);
    
    // Azimuth: Full rotation around listener (multiple rotations for variety)
    float theta = timeScaled * 4.0f * M_PI; // 2 full rotations
    
    // Elevation: Varies between up and down
    float phi = std::sin(timeScaled * 6.0f * M_PI) * M_PI / 2.0f; // From -90° to +90°
    
    // Distance: Varies between near (0.5m) and far (8.0m) for dramatic effect
    float minDistance = 0.5f;
    float maxDistance = 8.0f;
    // Use a different pattern for distance variation
    float distancePhase = std::fmod(timeScaled * 2.0f, 1.0f);
    float distance = minDistance + (maxDistance - minDistance) * 
                     (0.5f + 0.5f * std::sin(distancePhase * 2.0f * M_PI));
    
    // Convert to Cartesian coordinates
    // Steam Audio uses right-handed coordinate system:
    // +x = right, +y = up, +z = backward (opposite of forward)
    // Direction vector points FROM listener TO source
    
    IPLVector3 position;
    position.x = distance * std::cos(phi) * std::cos(theta);
    position.y = distance * std::sin(phi);
    position.z = -distance * std::cos(phi) * std::sin(theta); // Negative for forward direction
    
    // Calculate actual distance
    float actualDistance = std::sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
    
    // Normalize direction vector
    IPLVector3 direction;
    if (actualDistance > 0.0001f) {
        direction.x = position.x / actualDistance;
        direction.y = position.y / actualDistance;
        direction.z = position.z / actualDistance;
    } else {
        direction.x = 1.0f;
        direction.y = 0.0f;
        direction.z = 0.0f;
        actualDistance = 1.0f;
    }
    
    TrajectoryResult result;
    result.direction = direction;
    result.distance = actualDistance;
    return result;
}

int main(int argc, char* argv[]) {
    const std::string inputFile = "sample_music.wav";
    const std::string outputFile = "output_binaural.wav";
    
    // Parse command-line arguments
    bool useDefaultHRTF = false;
    bool exportTrajectory = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hrtf") {
            if (i + 1 < argc) {
                std::string hrtfType = argv[++i];
                if (hrtfType == "default") {
                    useDefaultHRTF = true;
                } else if (hrtfType == "custom") {
                    useDefaultHRTF = false;
                } else {
                    std::cerr << "Error: Unknown HRTF type: " << hrtfType << std::endl;
                    std::cerr << "Usage: " << argv[0] << " [--hrtf default|custom]" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --hrtf requires an argument (default or custom)" << std::endl;
                return 1;
            }
        } else if (arg == "--export-trajectory" || arg == "-e") {
            exportTrajectory = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --hrtf <default|custom>     Use default or custom HRTF (default: custom)" << std::endl;
            std::cout << "  --export-trajectory, -e     Export trajectory data to CSV file" << std::endl;
            std::cout << "  --help, -h                  Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Read input WAV file
    WAVFile inputWAV;
    if (!readWAV(inputFile, inputWAV)) {
        return 1;
    }

    // Initialize Steam Audio context
    IPLContext context = nullptr;
    IPLContextSettings contextSettings{};
    contextSettings.version = STEAMAUDIO_VERSION;
    contextSettings.simdLevel = IPL_SIMDLEVEL_NEON; // Use NEON on ARM64
    
    IPLerror error = iplContextCreate(&contextSettings, &context);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to create Steam Audio context" << std::endl;
        return 1;
    }
    std::cout << "Steam Audio context created\n";

    // Set up audio settings
    IPLAudioSettings audioSettings{};
    audioSettings.samplingRate = inputWAV.sampleRate;
    audioSettings.frameSize = 1024; // Process 1024 samples at a time

    // Create HRTF (default or custom SOFA)
    IPLHRTF hrtf = nullptr;
    IPLHRTFSettings hrtfSettings{};
    
    if (useDefaultHRTF) {
        // Use default HRTF
        hrtfSettings.type = IPL_HRTFTYPE_DEFAULT;
        hrtfSettings.volume = 1.0f;
        hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;
        
        std::cout << "Using default HRTF\n";
        
        error = iplHRTFCreate(context, &audioSettings, &hrtfSettings, &hrtf);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to create default HRTF" << std::endl;
            iplContextRelease(&context);
            return 1;
        }
        std::cout << "Default HRTF created\n";
    } else {
        // Use custom SOFA file
        hrtfSettings.type = IPL_HRTFTYPE_SOFA;
        
        // Use D2_HRIR_SOFA file matching the input sample rate
        std::string sofaFilePath;
        if (inputWAV.sampleRate == 44100) {
            sofaFilePath = "D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa";
        } else if (inputWAV.sampleRate == 48000) {
            sofaFilePath = "D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa";
        } else if (inputWAV.sampleRate == 96000) {
            sofaFilePath = "D2_HRIR_SOFA/D2_96K_24bit_512tap_FIR_SOFA.sofa";
        } else {
            std::cerr << fmt::format("Warning: Unsupported sample rate {} Hz. Using 48K SOFA file.\n", inputWAV.sampleRate);
            sofaFilePath = "D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa";
        }
        
        hrtfSettings.sofaFileName = sofaFilePath.c_str();
        hrtfSettings.sofaData = nullptr;
        hrtfSettings.sofaDataSize = 0;
        hrtfSettings.volume = 1.0f;
        hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;
        
        std::cout << fmt::format("Loading HRTF from SOFA file: {}\n", sofaFilePath);
        
        error = iplHRTFCreate(context, &audioSettings, &hrtfSettings, &hrtf);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: Failed to create HRTF from SOFA file: {}\n", sofaFilePath) << std::endl;
            std::cerr << "Make sure the SOFA file exists and is accessible." << std::endl;
            iplContextRelease(&context);
            return 1;
        }
        std::cout << "Custom HRTF created from SOFA file\n";
    }

    // Create binaural effect
    IPLBinauralEffect binauralEffect = nullptr;
    IPLBinauralEffectSettings effectSettings{};
    effectSettings.hrtf = hrtf;
    
    error = iplBinauralEffectCreate(context, &audioSettings, &effectSettings, &binauralEffect);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to create binaural effect" << std::endl;
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    std::cout << "Binaural effect created\n";

    // Set up distance attenuation model
    IPLDistanceAttenuationModel distanceModel{};
    distanceModel.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    distanceModel.minDistance = 0.0f; // No attenuation within 1 meter
    distanceModel.callback = nullptr;
    distanceModel.userData = nullptr;
    distanceModel.dirty = IPL_FALSE;
    
    // Listener is at origin (0, 0, 0)
    IPLVector3 listenerPosition = {0.0f, 0.0f, 0.0f};

    // Allocate audio buffers
    IPLAudioBuffer inputBuffer{};
    IPLAudioBuffer outputBuffer{};
    
    inputBuffer.numChannels = 1; // Mono input
    inputBuffer.numSamples = audioSettings.frameSize;
    error = iplAudioBufferAllocate(context, 1, audioSettings.frameSize, &inputBuffer);
    if (error != IPL_STATUS_SUCCESS || inputBuffer.data == nullptr) {
        std::cerr << "Error: Failed to allocate input buffer" << std::endl;
        iplBinauralEffectRelease(&binauralEffect);
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    
    outputBuffer.numChannels = 2; // Stereo output
    outputBuffer.numSamples = audioSettings.frameSize;
    error = iplAudioBufferAllocate(context, 2, audioSettings.frameSize, &outputBuffer);
    if (error != IPL_STATUS_SUCCESS || outputBuffer.data == nullptr) {
        std::cerr << "Error: Failed to allocate output buffer" << std::endl;
        iplAudioBufferFree(context, &inputBuffer);
        iplBinauralEffectRelease(&binauralEffect);
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    
    // Process audio in chunks
    std::vector<float> leftOutput;
    std::vector<float> rightOutput;
    
    size_t totalSamples = inputWAV.samples.size();
    float duration = static_cast<float>(totalSamples) / audioSettings.samplingRate;
    
    // Debug: Check input audio levels
    float maxInput = 0.0f;
    for (size_t i = 0; i < std::min(totalSamples, size_t(10000)); i++) {
        maxInput = std::max(maxInput, std::abs(inputWAV.samples[i]));
    }
    std::cout << fmt::format("Input audio peak level: {:.4f}\n", maxInput);
    
    std::cout << fmt::format("Processing {} samples ({:.2f} seconds)...\n", totalSamples, duration);
    
    // Trajectory data for export
    std::ofstream trajectoryFile;
    if (exportTrajectory) {
        trajectoryFile.open("trajectory.csv");
        if (trajectoryFile.is_open()) {
            trajectoryFile << "time,x,y,z,distance,azimuth,elevation\n";
            std::cout << "Exporting trajectory data to trajectory.csv\n";
        } else {
            std::cerr << "Warning: Could not open trajectory.csv for writing\n";
            exportTrajectory = false;
        }
    }
    
    float maxOutputLeft = 0.0f;
    float maxOutputRight = 0.0f;
    
    for (size_t offset = 0; offset < totalSamples; offset += audioSettings.frameSize) {
        size_t frameSize = std::min(static_cast<size_t>(audioSettings.frameSize), totalSamples - offset);
        
        // Fill input buffer (mono)
        for (size_t i = 0; i < frameSize; i++) {
            inputBuffer.data[0][i] = inputWAV.samples[offset + i];
        }
        // Zero-pad if needed
        for (size_t i = frameSize; i < audioSettings.frameSize; i++) {
            inputBuffer.data[0][i] = 0.0f;
        }
        
        // Zero output buffer
        for (size_t i = 0; i < audioSettings.frameSize; i++) {
            outputBuffer.data[0][i] = 0.0f;
            outputBuffer.data[1][i] = 0.0f;
        }
        
        // Calculate current position in trajectory
        float currentTime = static_cast<float>(offset) / audioSettings.samplingRate;
        TrajectoryResult trajectory = generateTrajectory(currentTime, duration);
        IPLVector3 direction = trajectory.direction;
        float distance = trajectory.distance;
        
        // Calculate distance attenuation
        // Source position is distance * direction from listener (which is at origin)
        IPLVector3 sourcePosition;
        sourcePosition.x = distance * direction.x;
        sourcePosition.y = distance * direction.y;
        sourcePosition.z = distance * direction.z;
        
        // Export trajectory data (sample every 10 frames to reduce file size)
        if (exportTrajectory && trajectoryFile.is_open() && (offset % (audioSettings.frameSize * 10) == 0)) {
            // Calculate azimuth and elevation for visualization
            float azimuth = std::atan2(sourcePosition.x, -sourcePosition.z) * 180.0f / M_PI;
            float elevation = std::asin(sourcePosition.y / distance) * 180.0f / M_PI;
            trajectoryFile << fmt::format("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.2f},{:.2f}\n",
                                         currentTime, sourcePosition.x, sourcePosition.y, sourcePosition.z,
                                         distance, azimuth, elevation);
        }
        
        IPLfloat32 distanceAttenuation = iplDistanceAttenuationCalculate(
            context, sourcePosition, listenerPosition, &distanceModel);
        
        // Apply distance attenuation to input audio
        for (size_t i = 0; i < audioSettings.frameSize; i++) {
            inputBuffer.data[0][i] *= distanceAttenuation;
        }
        
        // Set up binaural effect parameters
        IPLBinauralEffectParams params{};
        params.direction = direction;
        params.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
        params.spatialBlend = 1.0f; // Full spatialization
        params.hrtf = hrtf;
        
        // Apply binaural effect
        iplBinauralEffectApply(binauralEffect, &params, &inputBuffer, &outputBuffer);
        
        // Collect output samples and track max levels
        for (size_t i = 0; i < frameSize; i++) {
            leftOutput.push_back(outputBuffer.data[0][i]);
            rightOutput.push_back(outputBuffer.data[1][i]);
            maxOutputLeft = std::max(maxOutputLeft, std::abs(outputBuffer.data[0][i]));
            maxOutputRight = std::max(maxOutputRight, std::abs(outputBuffer.data[1][i]));
        }
        
        // Progress indicator
        if ((offset / audioSettings.frameSize) % 100 == 0) {
            float progress = (static_cast<float>(offset) / totalSamples) * 100.0f;
            std::cout << fmt::format("Progress: {:.1f}%\r", progress);
            std::cout.flush();
        }
    }
    
    std::cout << fmt::format("\nOutput audio peak levels - Left: {:.4f}, Right: {:.4f}\n", 
                             maxOutputLeft, maxOutputRight);
    
    // Close trajectory file
    if (exportTrajectory && trajectoryFile.is_open()) {
        trajectoryFile.close();
        std::cout << "Trajectory data exported to trajectory.csv\n";
    }
    
    // Process tail samples
    IPLAudioEffectState tailState = iplBinauralEffectGetTail(binauralEffect, &outputBuffer);
    while (tailState == IPL_AUDIOEFFECTSTATE_TAILREMAINING) {
        for (size_t i = 0; i < audioSettings.frameSize; i++) {
            leftOutput.push_back(outputBuffer.data[0][i]);
            rightOutput.push_back(outputBuffer.data[1][i]);
        }
        tailState = iplBinauralEffectGetTail(binauralEffect, &outputBuffer);
    }
    
    std::cout << "\nProcessing complete\n";

    // Free audio buffers
    iplAudioBufferFree(context, &inputBuffer);
    iplAudioBufferFree(context, &outputBuffer);

    // Write output WAV file
    if (!writeWAV(outputFile, leftOutput, rightOutput, audioSettings.samplingRate)) {
        iplBinauralEffectRelease(&binauralEffect);
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }

    // Cleanup
    iplBinauralEffectRelease(&binauralEffect);
    iplHRTFRelease(&hrtf);
    iplContextRelease(&context);

    std::cout << fmt::format("Successfully created binaural audio: {}\n", outputFile);
    return 0;
}
