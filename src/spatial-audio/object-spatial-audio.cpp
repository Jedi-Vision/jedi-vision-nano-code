#include <phonon.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>

using json = nlohmann::json;

// Constants
const int MAX_SIMULTANEOUS_OBJECTS = 10;
const int SAMPLE_RATE = 44100;
const int FRAME_SIZE = 1024;
const float VIDEO_FPS = 30.0f;
const float FRAME_INTERVAL_MS = 1000.0f / VIDEO_FPS; // 33.33ms
const float CAMERA_FOV_HORIZONTAL_DEG = 60.0f;
const float CAMERA_FOV_VERTICAL_DEG = 45.0f;
const float FADE_IN_TIME_MS = 100.0f;
const float FADE_OUT_TIME_MS = 100.0f;
const float PULSE_RATE_HZ = 2.0f;
const float BASE_FREQUENCY_HZ = 400.0f;
const float FREQUENCY_STEP_HZ = 100.0f;

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
        bool foundFmt = false;
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
                foundFmt = true;
                break;
            } else {
                file.seekg(chunkSize, std::ios::cur);
            }
        }
        
        if (!foundFmt) {
            std::cerr << "Error: Could not find fmt chunk in WAV file" << std::endl;
            file.close();
            return false;
        }
        
        // Find data chunk (there might be other chunks between fmt and data)
        while (file.read(chunkID, 4) && file.read(reinterpret_cast<char*>(&header.dataSize), 4)) {
            if (std::memcmp(chunkID, "data", 4) == 0) {
                break;
            } else {
                // Skip this chunk
                file.seekg(header.dataSize, std::ios::cur);
            }
        }
        
        if (std::memcmp(chunkID, "data", 4) != 0) {
            std::cerr << "Error: Could not find data chunk in WAV file" << std::endl;
            file.close();
            return false;
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

// Data structures for object detection
struct DetectedObject {
    int id;
    float x_2d;      // Normalized [0, 1]
    float y_2d;      // Normalized [0, 1]
    float depth;     // Depth in meters
    float confidence; // Optional confidence score
    std::string class_name; // Optional class name
};

struct DetectionFrame {
    int frame_number;
    float timestamp_ms;
    std::vector<DetectedObject> objects;
};

// Internal tracking state for interpolation
struct TrackedObject {
    int id;
    IPLVector3 current_position;
    IPLVector3 previous_position;
    float current_distance;
    float previous_distance;
    float fade_volume; // 0.0 to 1.0 for fade in/out
    bool active;
    uint64_t last_seen_timestamp_us;
    
    TrackedObject() : id(-1), current_distance(1.0f), previous_distance(1.0f), 
                     fade_volume(0.0f), active(false), last_seen_timestamp_us(0) {
        current_position = {0.0f, 0.0f, -1.0f};
        previous_position = {0.0f, 0.0f, -1.0f};
    }
};

// Convert 2D camera coordinates + depth to 3D Steam Audio coordinates
IPLVector3 convertToWorldSpace(float x_2d, float y_2d, float depth) {
    // Convert normalized coordinates [0, 1] to angles
    // x_2d: 0 = left, 1 = right
    // y_2d: 0 = top, 1 = bottom
    float normalized_x = x_2d - 0.5f; // [-0.5, 0.5]
    float normalized_y = 0.5f - y_2d; // [-0.5, 0.5] (invert Y)
    
    // Convert to radians
    float fov_h_rad = CAMERA_FOV_HORIZONTAL_DEG * M_PI / 180.0f;
    float fov_v_rad = CAMERA_FOV_VERTICAL_DEG * M_PI / 180.0f;
    
    // Calculate angles
    float horizontal_angle = normalized_x * fov_h_rad;
    float vertical_angle = normalized_y * fov_v_rad;
    
    // Convert to 3D position (camera-relative)
    // Steam Audio uses right-handed: x=right, y=up, z=backward
    IPLVector3 position;
    position.x = depth * std::tan(horizontal_angle);
    position.y = depth * std::tan(vertical_angle);
    position.z = -depth; // Forward in camera space, backward in Steam Audio
    
    return position;
}

// Object tracker for managing object lifecycle and interpolation
class ObjectTracker {
private:
    std::unordered_map<int, TrackedObject> tracked_objects;
    float fade_in_samples;
    float fade_out_samples;
    
public:
    ObjectTracker() {
        fade_in_samples = (FADE_IN_TIME_MS / 1000.0f) * SAMPLE_RATE;
        fade_out_samples = (FADE_OUT_TIME_MS / 1000.0f) * SAMPLE_RATE;
    }
    
    void updateFromFrame(const DetectionFrame& frame, uint64_t current_time_us) {
        // Mark all objects as potentially inactive
        for (auto& pair : tracked_objects) {
            pair.second.active = false;
        }
        
        // Update or create tracked objects
        for (const auto& obj : frame.objects) {
            auto it = tracked_objects.find(obj.id);
            if (it != tracked_objects.end()) {
                // Update existing object
                TrackedObject& tracked = it->second;
                tracked.previous_position = tracked.current_position;
                tracked.previous_distance = tracked.current_distance;
                tracked.current_position = convertToWorldSpace(obj.x_2d, obj.y_2d, obj.depth);
                tracked.current_distance = obj.depth;
                tracked.active = true;
                tracked.last_seen_timestamp_us = current_time_us;
                
                // Fade in if just appeared
                if (tracked.fade_volume < 1.0f) {
                    tracked.fade_volume = std::min(1.0f, tracked.fade_volume + 
                        (FRAME_SIZE / fade_in_samples));
                }
            } else {
                // New object
                TrackedObject tracked;
                tracked.id = obj.id;
                tracked.current_position = convertToWorldSpace(obj.x_2d, obj.y_2d, obj.depth);
                tracked.previous_position = tracked.current_position; // Start with same position
                tracked.current_distance = obj.depth;
                tracked.previous_distance = obj.depth;
                tracked.active = true;
                tracked.fade_volume = 0.0f; // Start faded out
                tracked.last_seen_timestamp_us = current_time_us;
                tracked_objects[obj.id] = tracked;
            }
        }
        
        // Handle fade out for disappeared objects
        for (auto& pair : tracked_objects) {
            if (!pair.second.active) {
                TrackedObject& tracked = pair.second;
                tracked.fade_volume = std::max(0.0f, tracked.fade_volume - 
                    (FRAME_SIZE / fade_out_samples));
            }
        }
    }
    
    // Get interpolated positions for current audio frame
    std::vector<std::pair<int, IPLVector3>> getInterpolatedPositions(
        float interpolation_factor) {
        std::vector<std::pair<int, IPLVector3>> result;
        
        for (auto& pair : tracked_objects) {
            TrackedObject& tracked = pair.second;
            
            // Skip if fully faded out
            if (tracked.fade_volume <= 0.0f) {
                continue;
            }
            
            // Linear interpolation between previous and current position
            IPLVector3 interpolated;
            interpolated.x = tracked.previous_position.x + 
                (tracked.current_position.x - tracked.previous_position.x) * interpolation_factor;
            interpolated.y = tracked.previous_position.y + 
                (tracked.current_position.y - tracked.previous_position.y) * interpolation_factor;
            interpolated.z = tracked.previous_position.z + 
                (tracked.current_position.z - tracked.previous_position.z) * interpolation_factor;
            
            result.push_back({tracked.id, interpolated});
        }
        
        // Limit to max simultaneous objects (prioritize by fade volume)
        if (result.size() > MAX_SIMULTANEOUS_OBJECTS) {
            std::sort(result.begin(), result.end(), 
                [this](const auto& a, const auto& b) {
                    return tracked_objects[a.first].fade_volume > 
                           tracked_objects[b.first].fade_volume;
                });
            result.resize(MAX_SIMULTANEOUS_OBJECTS);
        }
        
        return result;
    }
    
    float getFadeVolume(int object_id) const {
        auto it = tracked_objects.find(object_id);
        if (it != tracked_objects.end()) {
            return it->second.fade_volume;
        }
        return 0.0f;
    }
    
    float getDistance(int object_id) const {
        auto it = tracked_objects.find(object_id);
        if (it != tracked_objects.end()) {
            const TrackedObject& tracked = it->second;
            // Interpolate distance
            return tracked.previous_distance + 
                (tracked.current_distance - tracked.previous_distance) * 0.5f;
        }
        return 1.0f;
    }
    
    void updateInterpolationFactor(float factor) {
        // This could be used for more advanced interpolation
        // For now, we use linear interpolation in getInterpolatedPositions
    }
};

// Get audio samples from loaded WAV file for an object
// Handles resampling and looping if needed
void getAudioSamples(const WAVFile& wav, float* output, int num_samples, 
                     float start_time, float target_sample_rate) {
    if (wav.samples.empty()) {
        // No audio loaded, output silence
        for (int i = 0; i < num_samples; i++) {
            output[i] = 0.0f;
        }
        return;
    }
    
    float source_sample_rate = static_cast<float>(wav.sampleRate);
    size_t source_size = wav.samples.size();
    
    if (source_size == 0) {
        for (int i = 0; i < num_samples; i++) {
            output[i] = 0.0f;
        }
        return;
    }
    
    // If sample rates match, use direct indexing (no resampling needed)
    if (std::abs(source_sample_rate - target_sample_rate) < 1.0f) {
        size_t start_index = static_cast<size_t>(start_time * source_sample_rate) % source_size;
        for (int i = 0; i < num_samples; i++) {
            size_t index = (start_index + i) % source_size;
            output[i] = wav.samples[index];
        }
        return;
    }
    
    // Resample from source sample rate to target sample rate
    for (int i = 0; i < num_samples; i++) {
        // Calculate the time in the source audio
        float target_time = start_time + (static_cast<float>(i) / target_sample_rate);
        float source_sample_index = target_time * source_sample_rate;
        
        // Handle wrapping for looping
        float wrapped_index = std::fmod(source_sample_index, static_cast<float>(source_size));
        if (wrapped_index < 0.0f) {
            wrapped_index += source_size;
        }
        
        // Get the two samples for linear interpolation
        size_t source_index = static_cast<size_t>(wrapped_index);
        size_t next_index = (source_index + 1) % source_size;
        
        // Linear interpolation for smoother playback
        float fraction = wrapped_index - std::floor(wrapped_index);
        float sample1 = wav.samples[source_index];
        float sample2 = wav.samples[next_index];
        output[i] = sample1 + (sample2 - sample1) * fraction;
    }
}

// Read detection frames from JSON file
bool readDetectionFrames(const std::string& filename, std::vector<DetectionFrame>& frames) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }
        
        try {
            json j = json::parse(line);
            
            DetectionFrame frame;
            frame.frame_number = j.value("frame", line_number - 1);
            frame.timestamp_ms = j.value("timestamp_ms", frame.frame_number * FRAME_INTERVAL_MS);
            
            if (j.contains("objects") && j["objects"].is_array()) {
                for (const auto& obj_json : j["objects"]) {
                    DetectedObject obj;
                    obj.id = obj_json.value("id", -1);
                    obj.x_2d = obj_json.value("x", 0.5f);
                    obj.y_2d = obj_json.value("y", 0.5f);
                    obj.depth = obj_json.value("depth", 2.0f);
                    obj.confidence = obj_json.value("confidence", 1.0f);
                    obj.class_name = obj_json.value("class", "");
                    
                    if (obj.id >= 0) {
                        frame.objects.push_back(obj);
                    }
                }
            }
            
            frames.push_back(frame);
        } catch (const json::parse_error& e) {
            std::cerr << fmt::format("Warning: Failed to parse line {}: {}\n", 
                                    line_number, e.what());
        }
    }
    
    file.close();
    std::cout << fmt::format("Loaded {} detection frames\n", frames.size());
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

int main(int argc, char* argv[]) {
    const std::string inputFile = (argc > 1) ? argv[1] : "sample_detections.json";
    const std::string outputFile = (argc > 2) ? argv[2] : "output_spatial_objects.wav";
    const std::string audioFile = "sample_music.wav";
    
    // Parse command-line arguments
    bool useDefaultHRTF = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hrtf" && i + 1 < argc) {
            std::string hrtfType = argv[++i];
            useDefaultHRTF = (hrtfType == "default");
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [INPUT_JSON] [OUTPUT_WAV] [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --hrtf <default|custom>     Use default or custom HRTF (default: custom)" << std::endl;
            std::cout << "  --help, -h                  Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Load audio files for each object ID
    std::unordered_map<int, WAVFile> audioFiles;
    
    // Object ID 1 uses sample_music.wav
    std::cout << "Loading audio file for object ID 1: sample_music.wav\n";
    WAVFile audioWAV1;
    if (!readWAV("sample_music.wav", audioWAV1)) {
        std::cerr << "Error: Failed to load audio file for object ID 1: sample_music.wav" << std::endl;
        return 1;
    }
    audioFiles[1] = audioWAV1;
    std::cout << fmt::format("  Object ID 1: {} samples, {} Hz\n", audioWAV1.samples.size(), audioWAV1.sampleRate);
    
    // Object ID 2 uses sample_music_2.wav
    std::cout << "Loading audio file for object ID 2: sample_music_2.wav\n";
    WAVFile audioWAV2;
    if (!readWAV("sample_music_2.wav", audioWAV2)) {
        std::cerr << "Error: Failed to load audio file for object ID 2: sample_music_2.wav" << std::endl;
        return 1;
    }
    audioFiles[2] = audioWAV2;
    std::cout << fmt::format("  Object ID 2: {} samples, {} Hz\n", audioWAV2.samples.size(), audioWAV2.sampleRate);
    
    std::cout << fmt::format("Loaded audio files for {} object IDs\n", audioFiles.size());
    
    // Read detection frames
    std::vector<DetectionFrame> detection_frames;
    if (!readDetectionFrames(inputFile, detection_frames)) {
        return 1;
    }
    
    if (detection_frames.empty()) {
        std::cerr << "Error: No detection frames found in file" << std::endl;
        return 1;
    }
    
    // Calculate total duration
    float total_duration_ms = detection_frames.back().timestamp_ms + FRAME_INTERVAL_MS;
    float total_duration_sec = total_duration_ms / 1000.0f;
    size_t total_samples = static_cast<size_t>(total_duration_sec * SAMPLE_RATE);
    
    std::cout << fmt::format("Processing {} frames ({:.2f} seconds, {} samples)\n",
                            detection_frames.size(), total_duration_sec, total_samples);
    
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
    audioSettings.samplingRate = SAMPLE_RATE;
    audioSettings.frameSize = FRAME_SIZE;

    // Create HRTF
    IPLHRTF hrtf = nullptr;
    IPLHRTFSettings hrtfSettings{};
    
    if (useDefaultHRTF) {
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
        hrtfSettings.type = IPL_HRTFTYPE_SOFA;
        
        std::string sofaFilePath = "D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa";
        
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

    // Create binaural effects for each object (pool of MAX_SIMULTANEOUS_OBJECTS)
    std::vector<IPLBinauralEffect> binaural_effects(MAX_SIMULTANEOUS_OBJECTS, nullptr);
    IPLBinauralEffectSettings effectSettings{};
    effectSettings.hrtf = hrtf;
    
    for (int i = 0; i < MAX_SIMULTANEOUS_OBJECTS; i++) {
        error = iplBinauralEffectCreate(context, &audioSettings, &effectSettings, &binaural_effects[i]);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: Failed to create binaural effect {}\n", i) << std::endl;
            // Cleanup already created effects
            for (int j = 0; j < i; j++) {
                iplBinauralEffectRelease(&binaural_effects[j]);
            }
            iplHRTFRelease(&hrtf);
            iplContextRelease(&context);
            return 1;
        }
    }
    std::cout << fmt::format("Created {} binaural effects\n", MAX_SIMULTANEOUS_OBJECTS);

    // Set up distance attenuation model
    IPLDistanceAttenuationModel distanceModel{};
    distanceModel.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    distanceModel.minDistance = 1.0f;
    distanceModel.callback = nullptr;
    distanceModel.userData = nullptr;
    distanceModel.dirty = IPL_FALSE;
    
    // Listener is at origin (0, 0, 0)
    IPLVector3 listenerPosition = {0.0f, 0.0f, 0.0f};

    // Allocate audio buffers
    IPLAudioBuffer inputBuffer{};
    IPLAudioBuffer outputBuffer{};
    IPLAudioBuffer mixBuffer{};
    
    inputBuffer.numChannels = 1; // Mono input
    inputBuffer.numSamples = FRAME_SIZE;
    error = iplAudioBufferAllocate(context, 1, FRAME_SIZE, &inputBuffer);
    if (error != IPL_STATUS_SUCCESS || inputBuffer.data == nullptr) {
        std::cerr << "Error: Failed to allocate input buffer" << std::endl;
        for (auto& effect : binaural_effects) {
            iplBinauralEffectRelease(&effect);
        }
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    
    outputBuffer.numChannels = 2; // Stereo output
    outputBuffer.numSamples = FRAME_SIZE;
    error = iplAudioBufferAllocate(context, 2, FRAME_SIZE, &outputBuffer);
    if (error != IPL_STATUS_SUCCESS || outputBuffer.data == nullptr) {
        std::cerr << "Error: Failed to allocate output buffer" << std::endl;
        iplAudioBufferFree(context, &inputBuffer);
        for (auto& effect : binaural_effects) {
            iplBinauralEffectRelease(&effect);
        }
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    
    mixBuffer.numChannels = 2; // Stereo mix buffer
    mixBuffer.numSamples = FRAME_SIZE;
    error = iplAudioBufferAllocate(context, 2, FRAME_SIZE, &mixBuffer);
    if (error != IPL_STATUS_SUCCESS || mixBuffer.data == nullptr) {
        std::cerr << "Error: Failed to allocate mix buffer" << std::endl;
        iplAudioBufferFree(context, &inputBuffer);
        iplAudioBufferFree(context, &outputBuffer);
        for (auto& effect : binaural_effects) {
            iplBinauralEffectRelease(&effect);
        }
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }
    
    // Initialize object tracker
    ObjectTracker tracker;
    
    // Process audio
    std::vector<float> leftOutput;
    std::vector<float> rightOutput;
    
    size_t current_frame_idx = 0;
    float current_time_sec = 0.0f;
    float max_output_left = 0.0f;
    float max_output_right = 0.0f;
    
    std::cout << "Processing audio...\n";
    
    for (size_t sample_offset = 0; sample_offset < total_samples; sample_offset += FRAME_SIZE) {
        current_time_sec = static_cast<float>(sample_offset) / SAMPLE_RATE;
        float current_time_ms = current_time_sec * 1000.0f;
        
        // Update tracker with latest detection frame if needed
        while (current_frame_idx < detection_frames.size() && 
               detection_frames[current_frame_idx].timestamp_ms <= current_time_ms) {
            uint64_t timestamp_us = static_cast<uint64_t>(detection_frames[current_frame_idx].timestamp_ms * 1000.0f);
            tracker.updateFromFrame(detection_frames[current_frame_idx], timestamp_us);
            current_frame_idx++;
        }
        
        // Calculate interpolation factor for this audio frame
        // Interpolate between the last processed frame and the next frame
        float interpolation_factor = 0.0f; // Default: use last processed frame
        if (current_frame_idx > 0 && current_frame_idx < detection_frames.size()) {
            float frame_start_ms = detection_frames[current_frame_idx - 1].timestamp_ms;
            float frame_end_ms = detection_frames[current_frame_idx].timestamp_ms;
            if (frame_end_ms > frame_start_ms) {
                interpolation_factor = (current_time_ms - frame_start_ms) / (frame_end_ms - frame_start_ms);
                interpolation_factor = std::max(0.0f, std::min(1.0f, interpolation_factor)); // Clamp to [0, 1]
            }
        } else if (current_frame_idx >= detection_frames.size() && detection_frames.size() > 0) {
            // After last frame, use last frame position (no interpolation)
            interpolation_factor = 1.0f;
        }
        
        // Get interpolated positions for active objects
        auto active_objects = tracker.getInterpolatedPositions(interpolation_factor);
        
        // Zero mix buffer
        for (int i = 0; i < FRAME_SIZE; i++) {
            mixBuffer.data[0][i] = 0.0f;
            mixBuffer.data[1][i] = 0.0f;
        }
        
        // Process each active object
        for (size_t obj_idx = 0; obj_idx < active_objects.size() && obj_idx < MAX_SIMULTANEOUS_OBJECTS; obj_idx++) {
            int object_id = active_objects[obj_idx].first;
            IPLVector3 position = active_objects[obj_idx].second;
            
            // Calculate distance and direction
            float distance = std::sqrt(position.x * position.x + 
                                       position.y * position.y + 
                                       position.z * position.z);
            
            if (distance < 0.001f) {
                distance = 0.001f; // Avoid division by zero
            }
            
            IPLVector3 direction;
            direction.x = position.x / distance;
            direction.y = position.y / distance;
            direction.z = position.z / distance;
            
            // Get audio samples from the appropriate WAV file for this object
            auto audioIt = audioFiles.find(object_id);
            if (audioIt != audioFiles.end()) {
                getAudioSamples(audioIt->second, inputBuffer.data[0], FRAME_SIZE, current_time_sec, SAMPLE_RATE);
            } else {
                // No audio file for this object ID, output silence
                for (int i = 0; i < FRAME_SIZE; i++) {
                    inputBuffer.data[0][i] = 0.0f;
                }
            }
            
            // Apply fade volume
            float fade_vol = tracker.getFadeVolume(object_id);
            for (int i = 0; i < FRAME_SIZE; i++) {
                inputBuffer.data[0][i] *= fade_vol;
            }
            
            // Calculate distance attenuation
            IPLfloat32 distanceAttenuation = iplDistanceAttenuationCalculate(
                context, position, listenerPosition, &distanceModel);
            
            // Apply distance attenuation
            for (int i = 0; i < FRAME_SIZE; i++) {
                inputBuffer.data[0][i] *= distanceAttenuation;
            }
            
            // Zero output buffer
            for (int i = 0; i < FRAME_SIZE; i++) {
                outputBuffer.data[0][i] = 0.0f;
                outputBuffer.data[1][i] = 0.0f;
            }
            
            // Set up binaural effect parameters
            IPLBinauralEffectParams params{};
            params.direction = direction;
            params.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
            params.spatialBlend = 1.0f; // Full spatialization
            params.hrtf = hrtf;
            
            // Apply binaural effect
            iplBinauralEffectApply(binaural_effects[obj_idx], &params, &inputBuffer, &outputBuffer);
            
            // Mix into mix buffer
            for (int i = 0; i < FRAME_SIZE; i++) {
                mixBuffer.data[0][i] += outputBuffer.data[0][i];
                mixBuffer.data[1][i] += outputBuffer.data[1][i];
            }
        }
        
        // Collect output samples
        size_t frame_size = std::min(static_cast<size_t>(FRAME_SIZE), total_samples - sample_offset);
        for (size_t i = 0; i < frame_size; i++) {
            leftOutput.push_back(mixBuffer.data[0][i]);
            rightOutput.push_back(mixBuffer.data[1][i]);
            max_output_left = std::max(max_output_left, std::abs(mixBuffer.data[0][i]));
            max_output_right = std::max(max_output_right, std::abs(mixBuffer.data[1][i]));
        }
        
        // Progress indicator
        if ((sample_offset / FRAME_SIZE) % 100 == 0) {
            float progress = (static_cast<float>(sample_offset) / total_samples) * 100.0f;
            std::cout << fmt::format("Progress: {:.1f}%\r", progress);
            std::cout.flush();
        }
    }
    
    std::cout << fmt::format("\nOutput audio peak levels - Left: {:.4f}, Right: {:.4f}\n", 
                            max_output_left, max_output_right);
    
    // Process tail samples for all effects
    for (auto& effect : binaural_effects) {
        IPLAudioEffectState tailState = iplBinauralEffectGetTail(effect, &outputBuffer);
        while (tailState == IPL_AUDIOEFFECTSTATE_TAILREMAINING) {
            for (int i = 0; i < FRAME_SIZE; i++) {
                leftOutput.push_back(outputBuffer.data[0][i]);
                rightOutput.push_back(outputBuffer.data[1][i]);
            }
            tailState = iplBinauralEffectGetTail(effect, &outputBuffer);
        }
    }
    
    std::cout << "\nProcessing complete\n";

    // Free audio buffers
    iplAudioBufferFree(context, &inputBuffer);
    iplAudioBufferFree(context, &outputBuffer);
    iplAudioBufferFree(context, &mixBuffer);

    // Write output WAV file
    if (!writeWAV(outputFile, leftOutput, rightOutput, SAMPLE_RATE)) {
        for (auto& effect : binaural_effects) {
            iplBinauralEffectRelease(&effect);
        }
        iplHRTFRelease(&hrtf);
        iplContextRelease(&context);
        return 1;
    }

    // Cleanup
    for (auto& effect : binaural_effects) {
        iplBinauralEffectRelease(&effect);
    }
    iplHRTFRelease(&hrtf);
    iplContextRelease(&context);

    std::cout << fmt::format("Successfully created spatial audio: {}\n", outputFile);
    return 0;
}

