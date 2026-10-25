import React, { useState, useEffect } from "react";
import { Sliders, Save, CheckCircle, AlertCircle, RefreshCw } from "lucide-react";
import { fetchSettings, updateSettings } from "../services/api";
import { SystemSettingsData } from "../types";

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SystemSettingsData | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    fetchSettings()
      .then(setSettings)
      .catch((e) => console.error(e));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    setIsSaving(true);
    setSaveStatus(null);
    try {
      await updateSettings(settings);
      setSaveStatus("Settings successfully saved and applied to running pipeline.");
      setTimeout(() => setSaveStatus(null), 4000);
    } catch (err: any) {
      setSaveStatus(`Failed to persist settings: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  if (!settings) {
    return (
      <div className="p-8 text-center text-fleet-muted">
        Loading system configuration...
      </div>
    );
  }

  return (
    <div className="p-6 max-w-[1200px] mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-fleet-border pb-3">
        <div>
          <h2 className="text-[16px] font-bold text-fleet-text m-0">
            System Configuration & Detection Thresholds
          </h2>
          <p className="text-xs text-fleet-muted mt-0.5">
            Configure edge vision parameters, in-cab alerts, and inference settings
          </p>
        </div>

        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex items-center gap-1.5 px-4 py-2 bg-fleet-primary hover:bg-fleet-primaryHover text-white rounded text-xs font-bold transition-colors disabled:opacity-50"
        >
          <Save className="w-3.5 h-3.5" />
          <span>{isSaving ? "Saving..." : "Save & Apply Settings"}</span>
        </button>
      </div>

      {saveStatus && (
        <div
          className={`p-3 rounded-lg text-xs font-medium flex items-center gap-2 border ${
            saveStatus.includes("success")
              ? "bg-[#F0F8F3] text-fleet-normal border-[#D1E8D8]"
              : "bg-[#FDF2F2] text-fleet-critical border-[#F8D2D2]"
          }`}
        >
          {saveStatus.includes("success") ? (
            <CheckCircle className="w-4 h-4 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 shrink-0" />
          )}
          <span>{saveStatus}</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Detection Settings Section */}
        <div className="bg-white border border-fleet-border rounded-lg p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-fleet-text uppercase tracking-wider text-fleet-primary border-b border-fleet-border pb-2">
            1. Facial Telemetry & Detection Thresholds
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Eye Aspect Ratio (EAR) Threshold
              </label>
              <input
                type="number"
                step="0.01"
                min="0.10"
                max="0.35"
                value={settings.detection.ear_threshold}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      ear_threshold: parseFloat(e.target.value) || 0.21,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Initial benchmark starting point: 0.21. Individual calibration is recommended.
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Prolonged Eye Closure Duration (seconds)
              </label>
              <input
                type="number"
                step="0.1"
                min="0.5"
                max="5.0"
                value={settings.detection.eye_closure_duration_sec}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      eye_closure_duration_sec: parseFloat(e.target.value) || 1.2,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Elapsed closure time before triggering warning / critical state (default: 1.2s).
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Mouth Aspect Ratio (MAR) Yawn Threshold
              </label>
              <input
                type="number"
                step="0.05"
                min="0.40"
                max="0.95"
                value={settings.detection.mar_threshold}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      mar_threshold: parseFloat(e.target.value) || 0.65,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Threshold for oral aperture to distinguish yawning from standard speech (default: 0.65).
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Sustained Yawn Duration (seconds)
              </label>
              <input
                type="number"
                step="0.2"
                min="1.0"
                max="5.0"
                value={settings.detection.yawn_duration_sec}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      yawn_duration_sec: parseFloat(e.target.value) || 1.8,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Minimum continuous mouth opening to flag fatigue (default: 1.8s).
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Head Yaw Threshold (degrees)
              </label>
              <input
                type="number"
                step="1"
                min="15"
                max="60"
                value={settings.detection.head_yaw_threshold}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      head_yaw_threshold: parseFloat(e.target.value) || 28.0,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Max horizontal head rotation before flagging distraction (default: 28°).
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Distraction Duration Limit (seconds)
              </label>
              <input
                type="number"
                step="0.5"
                min="1.0"
                max="6.0"
                value={settings.detection.distraction_duration_sec}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    detection: {
                      ...settings.detection,
                      distraction_duration_sec: parseFloat(e.target.value) || 2.0,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Sustained head-turn limit before audio alert triggers (default: 2.0s).
              </span>
            </div>
          </div>
        </div>

        {/* Audio Advisory Settings */}
        <div className="bg-white border border-fleet-border rounded-lg p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-fleet-text uppercase tracking-wider text-fleet-primary border-b border-fleet-border pb-2">
            2. Audio Warning Policies & Alert Messages
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Alert Cooldown (seconds)
              </label>
              <input
                type="number"
                step="0.5"
                min="1.0"
                max="10.0"
                value={settings.audio.alert_cooldown_sec}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    audio: {
                      ...settings.audio,
                      alert_cooldown_sec: parseFloat(e.target.value) || 3.5,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg"
              />
              <span className="text-[11px] text-fleet-muted">
                Prevents overlapping alarms on successive video frames.
              </span>
            </div>

            <div>
              <label className="font-semibold text-fleet-text block mb-1">
                Master Audio System Toggle
              </label>
              <div className="pt-2">
                <label className="flex items-center gap-2 cursor-pointer font-medium text-fleet-text">
                  <input
                    type="checkbox"
                    checked={settings.audio.enable_audio}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        audio: {
                          ...settings.audio,
                          enable_audio: e.target.checked,
                        },
                      })
                    }
                    className="rounded border-fleet-border text-fleet-primary focus:ring-0"
                  />
                  <span>Enable In-Cab Audio Alerts</span>
                </label>
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="font-semibold text-fleet-text block mb-1">
                Level 1 Voice Message (Distraction / Head Turn)
              </label>
              <input
                type="text"
                value={settings.audio.level_1_message}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    audio: {
                      ...settings.audio,
                      level_1_message: e.target.value,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg font-mono"
              />
            </div>

            <div className="md:col-span-2">
              <label className="font-semibold text-fleet-text block mb-1">
                Level 2 Voice Message (Drowsiness / Fatigue)
              </label>
              <input
                type="text"
                value={settings.audio.level_2_message}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    audio: {
                      ...settings.audio,
                      level_2_message: e.target.value,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg font-mono"
              />
            </div>

            <div className="md:col-span-2">
              <label className="font-semibold text-fleet-text block mb-1">
                Level 3 Voice Message (Critical Prolonged Eye Closure)
              </label>
              <input
                type="text"
                value={settings.audio.level_3_message}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    audio: {
                      ...settings.audio,
                      level_3_message: e.target.value,
                    },
                  })
                }
                className="w-full border border-fleet-border rounded p-2 bg-fleet-bg font-mono"
              />
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};
