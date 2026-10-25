"""Safety event recording, filtering, and CSV export service."""
import csv
import io
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SafetyEvent(BaseModel):
    id: str
    timestamp: str
    epoch_time: float
    vehicle_id: str
    camera_id: str
    event_type: str        # "PROLONGED_CLOSURE", "DROWSY", "DISTRACTED", "YAWN", "PHONE_DETECTED", "FACE_LOST"
    severity: str          # "NORMAL", "WARNING", "CRITICAL"
    duration_sec: float
    confidence: Optional[float] = None
    description: str


class EventService:
    def __init__(self, max_stored_events: int = 2000):
        self.max_stored_events = max_stored_events
        self.events: List[SafetyEvent] = []
        self._seed_recent_initial_events()

    def _seed_recent_initial_events(self):
        """Seed a few realistic historical events for operational demo context."""
        now = time.time()
        sample_records = [
            ("FL-101", "Cam-0", "DISTRACTED", "WARNING", 2.3, 0.91, "Driver sustained head glance to passenger side", now - 3600 * 2.5),
            ("FL-101", "Cam-0", "YAWN", "WARNING", 2.1, 0.88, "Driver sustained yawning event detected", now - 3600 * 1.8),
            ("FL-101", "Cam-0", "PROLONGED_CLOSURE", "CRITICAL", 1.9, 0.94, "Prolonged eye closure (>1.5s) triggered audio warning", now - 3600 * 0.9),
            ("FL-101", "Cam-0", "DROWSY", "CRITICAL", 2.4, 0.92, "MobileNetV3 classified drowsy facial state", now - 3600 * 0.4),
        ]
        for v_id, c_id, e_type, sev, dur, conf, desc, ts in sample_records:
            dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            self.events.append(SafetyEvent(
                id=f"EVT-{int(ts)}",
                timestamp=dt_str,
                epoch_time=ts,
                vehicle_id=v_id,
                camera_id=c_id,
                event_type=e_type,
                severity=sev,
                duration_sec=dur,
                confidence=conf,
                description=desc
            ))

    def record_event(self,
                     event_type: str,
                     severity: str,
                     duration_sec: float,
                     confidence: Optional[float] = None,
                     description: str = "",
                     vehicle_id: str = "LOCAL-TRUCK-01",
                     camera_id: str = "Camera-0") -> SafetyEvent:
        now = time.time()
        evt_id = f"EVT-{int(now * 1000)}"
        dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        event = SafetyEvent(
            id=evt_id,
            timestamp=dt_str,
            epoch_time=now,
            vehicle_id=vehicle_id,
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
            duration_sec=round(duration_sec, 2),
            confidence=round(confidence, 2) if confidence is not None else None,
            description=description
        )

        self.events.insert(0, event)
        if len(self.events) > self.max_stored_events:
            self.events.pop()
        return event

    def get_events(self,
                   severity: Optional[str] = None,
                   event_type: Optional[str] = None,
                   search: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        filtered = self.events

        if severity and severity.upper() != "ALL":
            filtered = [e for e in filtered if e.severity.upper() == severity.upper()]

        if event_type and event_type.upper() != "ALL":
            filtered = [e for e in filtered if e.event_type.upper() == event_type.upper()]

        if search:
            s = search.lower()
            filtered = [e for e in filtered if (
                s in e.description.lower() or
                s in e.event_type.lower() or
                s in e.vehicle_id.lower() or
                s in e.id.lower()
            )]

        return [e.model_dump() for e in filtered[:limit]]

    def export_csv(self) -> str:
        """Exports all recorded safety events to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Event ID", "Timestamp", "Vehicle ID", "Camera ID", "Event Type", "Severity", "Duration (s)", "Confidence", "Description"])
        for e in self.events:
            writer.writerow([
                e.id,
                e.timestamp,
                e.vehicle_id,
                e.camera_id,
                e.event_type,
                e.severity,
                e.duration_sec,
                e.confidence if e.confidence is not None else "N/A",
                e.description
            ])
        return output.getvalue()

    def get_today_summary(self) -> Dict[str, Any]:
        """Calculates operational KPIs for overview page."""
        now = time.time()
        day_start = now - (now % 86400)
        today_events = [e for e in self.events if e.epoch_time >= day_start]

        critical_count = sum(1 for e in today_events if e.severity == "CRITICAL")
        warning_count = sum(1 for e in today_events if e.severity == "WARNING")
        drowsy_count = sum(1 for e in today_events if e.event_type in ["DROWSY", "PROLONGED_CLOSURE"])
        distracted_count = sum(1 for e in today_events if e.event_type in ["DISTRACTED", "PHONE_DETECTED"])

        return {
            "total_today": len(today_events),
            "critical_today": critical_count,
            "warning_today": warning_count,
            "drowsy_events": drowsy_count,
            "distraction_events": distracted_count,
            "last_event": self.events[0].model_dump() if self.events else None
        }
