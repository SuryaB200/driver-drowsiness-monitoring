import { useState, useRef, useCallback } from "react";
import { AudioAlertPayload } from "../types";

export function useDriverAudioAlert(initialEnabled = true, initialVolume = 1.0) {
  const [audioEnabled, setAudioEnabled] = useState(initialEnabled);
  const [volume, setVolume] = useState(initialVolume);
  const [lastSpokenMessage, setLastSpokenMessage] = useState<string | null>(null);
  const [lastSpokenTime, setLastSpokenTime] = useState<number>(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const isSpeakingRef = useRef<boolean>(false);
  const lastAlertTimeRef = useRef<number>(0);

  // Initialize Web Audio Context on first interaction
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        audioContextRef.current = new AudioCtx();
      }
    }
    if (audioContextRef.current && audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  // Synthetic tone chime
  const playChime = useCallback((freq = 880, duration = 0.25) => {
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, ctx.currentTime);

      gain.gain.setValueAtTime(volume * 0.4, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      // Audio context might be restricted
    }
  }, [getAudioContext, volume]);

  const speakAlert = useCallback((message: string, priorityLevel = 1) => {
    if (!audioEnabled || !("speechSynthesis" in window)) return;

    // Critical warnings (Level 3) cancel any ongoing lower-priority announcements
    if (priorityLevel === 3) {
      window.speechSynthesis.cancel();
      isSpeakingRef.current = false;
      playChime(950, 0.4);
    } else {
      playChime(660, 0.2);
    }

    if (isSpeakingRef.current && priorityLevel < 3) {
      return; // Prevent overlapping speech
    }

    try {
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.volume = volume;
      utterance.rate = 1.05; // Slightly urgent, clear cadence
      utterance.pitch = priorityLevel === 3 ? 1.2 : 1.0;

      utterance.onstart = () => {
        isSpeakingRef.current = true;
        setLastSpokenMessage(message);
        setLastSpokenTime(Date.now());
      };

      utterance.onend = () => {
        isSpeakingRef.current = false;
      };

      utterance.onerror = () => {
        isSpeakingRef.current = false;
      };

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      isSpeakingRef.current = false;
    }
  }, [audioEnabled, volume, playChime]);

  const handleAudioAlert = useCallback((payload: AudioAlertPayload | null, cooldownSec = 3.5) => {
    if (!payload || !payload.should_play || !audioEnabled) return;

    const now = Date.now();
    const elapsedSec = (now - lastAlertTimeRef.current) / 1000.0;

    // Critical level 3 bypasses cooldown if previous was lower level
    if (elapsedSec >= cooldownSec || payload.level === 3) {
      lastAlertTimeRef.current = now;
      speakAlert(payload.message, payload.level);
    }
  }, [audioEnabled, speakAlert]);

  const stopAllAudio = useCallback(() => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    isSpeakingRef.current = false;
  }, []);

  return {
    audioEnabled,
    setAudioEnabled,
    volume,
    setVolume,
    lastSpokenMessage,
    lastSpokenTime,
    handleAudioAlert,
    speakAlert,
    playChime,
    stopAllAudio,
  };
}
