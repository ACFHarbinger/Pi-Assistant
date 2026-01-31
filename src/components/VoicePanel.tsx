import React, { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Mic, MicOff, Radio, Loader2 } from 'lucide-react';

type VoiceStatus = 'idle' | 'listening' | 'recording' | 'processing';

export const VoicePanel: React.FC = () => {
    const [isListening, setIsListening] = useState(false);
    const [status, setStatus] = useState<VoiceStatus>('idle');
    const [lastTranscription, setLastTranscription] = useState<string | null>(null);

    const toggleListener = async () => {
        try {
            if (isListening) {
                await invoke('stop_voice_listener');
                setIsListening(false);
                setStatus('idle');
            } else {
                await invoke('start_voice_listener');
                setIsListening(true);
                setStatus('listening');
            }
        } catch (error) {
            console.error('Failed to toggle voice listener:', error);
        }
    };

    const handlePttDown = useCallback(async () => {
        if (status === 'recording' || status === 'processing') return;
        try {
            setStatus('recording');
            setLastTranscription(null);
            await invoke('push_to_talk_start');
        } catch (error) {
            console.error('Failed to start push-to-talk:', error);
            setStatus(isListening ? 'listening' : 'idle');
        }
    }, [status, isListening]);

    const handlePttUp = useCallback(async () => {
        if (status !== 'recording') return;
        try {
            setStatus('processing');
            const transcription = await invoke<string>('push_to_talk_stop');
            setLastTranscription(transcription);
            setStatus(isListening ? 'listening' : 'idle');
        } catch (error) {
            console.error('Failed to stop push-to-talk:', error);
            setStatus(isListening ? 'listening' : 'idle');
        }
    }, [status, isListening]);

    const statusColor = () => {
        switch (status) {
            case 'listening': return 'bg-red-500/20 border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.3)]';
            case 'recording': return 'bg-orange-500/20 border-orange-500/50 shadow-[0_0_20px_rgba(249,115,22,0.3)]';
            case 'processing': return 'bg-blue-500/20 border-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.3)]';
            default: return 'bg-white/10 border-white/20 shadow-xl';
        }
    };

    const statusText = () => {
        switch (status) {
            case 'listening': return 'Listening...';
            case 'recording': return 'Recording...';
            case 'processing': return 'Transcribing...';
            default: return 'Off';
        }
    };

    return (
        <div className="fixed bottom-24 right-8 z-50">
            <div className={`
                flex items-center gap-3 p-3 rounded-2xl
                backdrop-blur-xl border transition-all duration-300
                ${statusColor()}
            `}>
                {(status === 'listening' || status === 'recording') && (
                    <div className="flex gap-1 px-2">
                        {[1, 2, 3].map((i) => (
                            <div
                                key={i}
                                className={`w-1 h-4 rounded-full animate-pulse ${
                                    status === 'recording' ? 'bg-orange-400' : 'bg-red-400'
                                }`}
                                style={{ animationDelay: `${i * 0.2}s` }}
                            />
                        ))}
                    </div>
                )}

                {status === 'processing' && (
                    <div className="px-2">
                        <Loader2 size={16} className="text-blue-400 animate-spin" />
                    </div>
                )}

                {/* Wake word toggle */}
                <button
                    onClick={toggleListener}
                    disabled={status === 'recording' || status === 'processing'}
                    className={`
                        p-3 rounded-xl transition-all duration-300
                        ${isListening
                            ? 'bg-red-500 text-white hover:bg-red-600'
                            : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-500/30'}
                        disabled:opacity-50 disabled:cursor-not-allowed
                    `}
                    title={isListening ? 'Stop wake word detection' : 'Start wake word detection'}
                >
                    {isListening ? <Mic size={20} /> : <MicOff size={20} />}
                </button>

                {/* Push-to-talk button */}
                <button
                    onMouseDown={handlePttDown}
                    onMouseUp={handlePttUp}
                    onMouseLeave={status === 'recording' ? handlePttUp : undefined}
                    onTouchStart={handlePttDown}
                    onTouchEnd={handlePttUp}
                    disabled={status === 'processing'}
                    className={`
                        p-3 rounded-xl transition-all duration-300
                        ${status === 'recording'
                            ? 'bg-orange-500 text-white scale-110 shadow-lg shadow-orange-500/40'
                            : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-lg shadow-emerald-500/30'}
                        disabled:opacity-50 disabled:cursor-not-allowed
                    `}
                    title="Hold to talk"
                >
                    <Radio size={20} />
                </button>

                <div className="flex flex-col pr-2">
                    <span className="text-xs font-medium text-white/50 uppercase tracking-wider">
                        Voice Mode
                    </span>
                    <span className="text-sm font-semibold text-white">
                        {statusText()}
                    </span>
                    {lastTranscription && status === 'idle' && (
                        <span className="text-xs text-white/40 truncate max-w-[120px]" title={lastTranscription}>
                            {lastTranscription}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};
