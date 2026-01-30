"""
The point of this file is to create a SystemManagment class that can:
    track system interface properties (volume, temperature, etc)
    Store metrics in memory
    Write JSON log lines to a log file of the current state of the interface properties as well as the metrics


One main goal is to provide a simple decorator that other modules can use to:
    Automtically record when a function starts and ends
    Record the duration of the function
    Record the return value of the function             TODO
    Record any exception raised by the function         TODO




Typical usage from another module (Import and call the decorator on a function):

    from system_management import sys_mgmt, log_block

    @log_block("audio_info_to_spatial_audio")
    def run_audio_step(tokens_per_sec: float):
        # Update live system interface fields
        sys_mgmt.tokensPerSec = tokens_per_sec

        # Log a specific metric for later analysis
        sys_mgmt.logMetric("audio.tokensPerSec", tokens_per_sec)


"""

from dataclasses import (
    dataclass,
    field,
)  # for small record-like classes: Metric, ComponentState
from datetime import datetime, timedelta  # for timestamps and uptime durations
from enum import Enum  # for ChargeState and LogFormat enums           TODO
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
)  # for type hints and annotations                TODO remove?

import logging  # built-in logging framework, used to write JSON logs to a file
import json  # convert Python dicts -> JSON strings for logs and export
import time  # for low-level timing (uptime and duration of functions)
import os  # for creating directories

import psutil  # for getting system metrics (CPU usage, memory usage, etc)


# --------------------------------------------------------------------------------------
# Basic data structures and enums (ChargeState, LogFormat, Metric, ComponentState)
# --------------------------------------------------------------------------------------


# ChargeState will be used to represent the state of the battery (assumes we use a battery still and don't pivot)
class ChargeState(str, Enum):
    CHARGING = "Charging"
    DISCHARGING = "Discharging"
    FULL = "Full"
    NOT_CHARGING = "NotCharging"
    UNKNOWN = "Unknown"


# LogFormat will be used to represent the format of the output log file (default is JSON)
class LogFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    BINARY = "binary"

# Metric is a simple record-like class that represents a single data point.
# It has a name, a value, and a timestamp. The @dataclass decorator automatically creates the __init__ method 
@dataclass
class Metric:
    """
    A single metric data point.

    Example:
        name="audio.tokensPerSec",
        value=73.2,
        timestamp=<when the metric was recorded>
    """

    name: str
    value: Any
    timestamp: datetime

# ComponentState is a simple record-like class that represents the state of a single component (block) in the system (audio, env_representation, etc.)
@dataclass
class ComponentState:
    """
    Represents the state of a single component (block) in the system.

    This matches the UML ComponentState class.
    """

    id: str  # component identifier (e.g., "env_representation")
    status: str  # free-form status string (e.g., "OK", "DEGRADED", "ERROR")
    outputs: Dict[str, Any] = field(
        default_factory=dict
    )  # last outputs from that component
    lastUpdated: datetime = field(
        default_factory=datetime.utcnow
    )  # last time this state was updated


# This Protocol is basically a tempalte for applying settings (segment specific objects, throttle FPS, etc) for others to implement throughout the system
class SettingsConsumer(Protocol):
    """
    Protocol (interface) for objects that can receive settings.

    This corresponds to the SettingsConsumer interface in the UML.
    Any class that implements applySettings(settings: Dict[str, Any]) is a SettingsConsumer.
    """

    # def applySettings(self, settings: Dict[str, Any]) -> None:  # pragma: no cover - just an interface
    #     ...


# --------------------------------------------------------------------------------------
# Logging backend setup: create logs directory, configure JSON-line logging
# --------------------------------------------------------------------------------------
# Ensure the "logs" directory exists so FileHandler doesn't crash
os.makedirs("logs", exist_ok=True)

# Log formatter that encodes a dict as JSON
class JsonFormatter(logging.Formatter):
    """
    Simple formatter that expects 'record.msg' to be a dict and JSON-encodes it.

    This keeps log writing super simple: we just pass a dict when calling logger.info().
    If someone logs a plain string by mistake, we wrap it in {"msg": "..."}.
    """

    def format(self, record: logging.LogRecord) -> str:
        # If msg is already a dict, use it as-is; otherwise, wrap it in a dict
        payload = (
            record.msg if isinstance(record.msg, dict) else {"msg": record.getMessage()}
        )
        return json.dumps(payload)


# Create a module-level logger instance that writes to logs/system.log
_logger = logging.getLogger("jedi_sys")
_logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler("logs/system.log")
_file_handler.setFormatter(JsonFormatter())
_logger.addHandler(_file_handler)


# --------------------------------------------------------------------------------------
# SystemManagement singleton implementation
# --------------------------------------------------------------------------------------

# Singleton\ class to ensure only one instance in application (multiple instances would mess up logging data)
class SystemManagement:
    """
    Central singleton that holds:

    - System Interface Properties (volume, battery percent, etc.)
    - State & Settings (settings map, componentStates map, in-memory metrics list)
    - Operations to log metrics and export logs

    In other modules you should NOT instantiate this directly.
    Instead, import the singleton instance 'sys_mgmt' defined at the bottom of this file.
    """

    # This class variable stores the single allowed instance
    _instance: Optional["SystemManagement"] = None

    def __init__(self) -> None:
        """
        Constructor is kept "private" by convention:
        you should use SystemManagement.instance() instead of calling SystemManagement().
        """

        # --------------------------
        # System Interface Properties (from UML)
        # --------------------------

        # All of these map directly to required interface properties names (see _interface_snapshot).
        self.volume: int = 50  # 0..100, default mid-volume
        self.deviceTempPercent: int = (
            0  # device temperature as percent of some max (0..100)
        )
        self.chargeState: ChargeState = ChargeState.UNKNOWN  # current charging state
        self.batteryTempCelsius: float = 0.0  # battery temperature in degrees Celsius
        self.batteryPercent: int = 0  # 0..100
        self.tokensPerSec: float = 0.0  # recent model throughput
        self.deviceUptime: timedelta = timedelta(
            seconds=0
        )  # will be recomputed on the fly
        self.voiceCommandButtonPressed: bool = (
            False  # current state of voice command button
        )
        self.lastFrameGroupProcessed: str = (
            ""  # last frame group ID that completed processing
        )

        # --------------------------
        # State & Settings (from UML)
        # --------------------------

        # Arbitrary key-value settings
        self.settings: Dict[str, Any] = {}

        # Component ID -> ComponentState
        self.componentStates: Dict[str, ComponentState] = {}

        # In-memory list of metrics (useful for queries and export)
        self.logs: List[Metric] = []

        # Internal start time used to compute uptime in seconds
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "SystemManagement":
        """
        Returns the single SystemManagement instance.

        Usage:
            sys_mgmt = SystemManagement.instance()
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal helper: snapshot of interface properties with required names
    # ------------------------------------------------------------------

    #  Captures ALL system properties at a moment in time
    def _interface_snapshot(self) -> Dict[str, Any]:
        """
        Build a dict containing all required System Interface Properties
        using the exact names from interface spec:

            sysmgmt_all_volume_int
            sysmgmt_all_timestamp_clock
            sysmgmt_all_deviceTemp_percent
            sysmgmt_all_chargeState_enum
            sysmgmt_all_batteryTemp_celsius
            sysmgmt_all_batteryPercent_percent
            sysmgmt_all_tokensPerSec_float
            sysmgmt_all_deviceUptime_seconds
            sysmgmt_all_voiceButton_state

        This snapshot is attached to every log record so that
        each metric can be correlated with system status at that time.
        """

        now = datetime.utcnow()
        uptime_seconds = int(time.time() - self._start_time)
        self.deviceUptime = timedelta(seconds=uptime_seconds)

        return {
            "sysmgmt_all_volume_int": int(self.volume),
            "sysmgmt_all_timestamp_clock": now.isoformat(timespec="milliseconds") + "Z",
            "sysmgmt_all_deviceTemp_percent": int(self.deviceTempPercent),
            "sysmgmt_all_chargeState_enum": self.chargeState.value,
            "sysmgmt_all_batteryTemp_celsius": float(self.batteryTempCelsius),
            "sysmgmt_all_batteryPercent_percent": int(self.batteryPercent),
            "sysmgmt_all_tokensPerSec_float": float(self.tokensPerSec),
            "sysmgmt_all_deviceUptime_seconds": uptime_seconds,
            "sysmgmt_all_voiceButton_state": bool(self.voiceCommandButtonPressed),
        }

    # ------------------------------------------------------------------
    # Operations: metrics & logs (matching required UML names)
    # ------------------------------------------------------------------

    def logMetric(
        self, name: str, value: Any, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a new metric in memory AND append it to the log file.

        Parameters:
            name:     Logical name of the metric (e.g. "audio.tokensPerSec")
            value:    Any Python value that can be JSON-serialized
            timestamp: Optional timestamp; if None, uses current UTC time

        This is the main method other modules will call.
        """

        ts = timestamp or datetime.utcnow()

        # Store in memory for later queries/export
        metric = Metric(name=name, value=value, timestamp=ts)
        self.logs.append(metric)

        # Build the log record payload (metric fields + interface snapshot)
        record = {
            "type": "metric",
            "name": name,
            "value": value,
            "timestamp": ts.isoformat(timespec="milliseconds") + "Z",
            **self._interface_snapshot(),  # expands all sysmgmt_all_* properties
        }

        # Write to logs/system.log as a single JSON line
        _logger.info(record)

    def getMetrics(self, since: Optional[datetime] = None) -> List[Metric]:
        """
        Return all metrics, optionally filtering by 'since' timestamp.

        This does NOT read from the log file; it just returns what is currently
        stored in memory in self.logs.
        """
        if since is None:
            return list(self.logs)
        return [m for m in self.logs if m.timestamp >= since]

    def exportLogs(self, format: LogFormat = LogFormat.JSON) -> str:
        """
        Export in-memory metrics in the requested format.

        Currently only JSON is implemented; CSV/BINARY could be added later
        for extra credit or future work.
        """
        if format is LogFormat.JSON:
            # Convert Metric objects into plain dicts for JSON serialization
            return json.dumps(
                [m.__dict__ for m in self.logs],
                default=str,  # ensures datetime is converted to string
            )

        # This keeps behavior explicit: other formats are "future work"
        raise NotImplementedError(f"{format} export not implemented")

    # ------------------------------------------------------------------
    # Operations: component state & settings (matching UML)
    # ------------------------------------------------------------------

    def setComponentState(self, id: str, state: ComponentState) -> None:
        """
        Store or update the state of a component.

        Typical usage:
            sys_mgmt.setComponentState(
                "env_representation",
                ComponentState(id="env_representation", status="OK")
            )
        """
        state.lastUpdated = datetime.utcnow()
        self.componentStates[id] = state

    def getComponentState(self, id: str) -> Optional[ComponentState]:
        """
        Retrieve the state of a component by id.
        Returns None if the component is unknown.
        """
        return self.componentStates.get(id)

    def updateSetting(self, key: str, value: Any) -> None:
        """
        Update or create a setting.

        You can use this for tuning parameters that might be changed at runtime
        (e.g., volume, threshold values, model selection).
        """
        self.settings[key] = value

    def getSetting(self, key: str) -> Any:
        """
        Get a setting value by key. Returns None if the key does not exist.
        """
        return self.settings.get(key)

    def pushSettingsTo(self, consumer: SettingsConsumer) -> None:
        """
        Push current settings to a SettingsConsumer.

        Any object with an applySettings(settings: Dict[str, Any]) method
        can receive the entire settings dict.
        """
        consumer.applySettings(self.settings)

    # ------------------------------------------------------------------
    # Helper operations: frame groups, bottleneck analysis (simple version)
    # ------------------------------------------------------------------

    def recordFrameGroupProcessed(
        self, groupId: str, ts: Optional[datetime] = None
    ) -> None:
        """
        Record that a frame group has been processed.

        Updates lastFrameGroupProcessed and logs a metric.
        """
        self.lastFrameGroupProcessed = groupId
        self.logMetric("lastFrameGroupProcessed", groupId, timestamp=ts)

    def findBottlenecks(self, windowSec: int = 60) -> List[str]:
        """
        VERY SIMPLE example of a bottleneck finder.

        Looks at metrics from the last 'windowSec' seconds and returns a list
        of metric names whose average value is particularly "large".

        You can replace this logic with something more sophisticated later.
        """
        cutoff = datetime.utcnow() - timedelta(seconds=windowSec)
        recent = [m for m in self.logs if m.timestamp >= cutoff]

        # Compute average per metric name
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for m in recent:
            # Only consider numeric values for this simple example
            if isinstance(m.value, (int, float)):
                sums[m.name] = sums.get(m.name, 0.0) + float(m.value)
                counts[m.name] = counts.get(m.name, 0) + 1

        averages = {
            name: (sums[name] / counts[name]) for name in sums if counts[name] > 0
        }

        # In a real system, you'd compare against known thresholds.
        # Here, we'll just return the top 5 highest averages as "bottlenecks".
        sorted_names = sorted(averages, key=averages.get, reverse=True)
        return sorted_names[:5]


# --------------------------------------------------------------------------------------
# Convenience: module-level singleton and decorator for easy usage elsewhere
# --------------------------------------------------------------------------------------

# This is the instance you will actually import and use in other modules.
sys_mgmt: SystemManagement = SystemManagement.instance()


def log_block(block_name: str):
    """
    Decorator that logs the start and end of a function, including its duration.

    Example usage:

        from system_management import log_block

        @log_block("env_representation")
        def run_env_representation():
            # ... work ...

    This will automatically create metrics:
        "env_representation.start"  (True)
        "env_representation.duration_ms"  (<milliseconds>)
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.time()

            # Log a "start" metric (you can use it for debugging timelines)
            sys_mgmt.logMetric(f"{block_name}.start", True)

            try:
                return fn(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000.0
                sys_mgmt.logMetric(f"{block_name}.duration_ms", duration_ms)

        return wrapper

    return decorator
