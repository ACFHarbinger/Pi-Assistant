import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Mic, MicOff, Volume2, Loader2 } from 'lucide-react';

export const VoicePanel: React.FC = () => {
    const [isListening, setIsListening] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [status, setStatus] = useState<'idle' | 'listening' | 'processing'>('idle');

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

    return (
        <div className="fixed bottom-24 right-8 z-50">
            <div className={`
        flex items-center gap-3 p-3 rounded-2xl
        backdrop-blur-xl border transition-all duration-300
        ${status === 'listening'
                    ? 'bg-red-500/20 border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.3)]'
                    : 'bg-white/10 border-white/20 shadow-xl'}
      `}>
                {status === 'listening' && (
                    <div className="flex gap-1 px-2">
                        {[1, 2, 3].map((i) => (
                            <div
                                key={i}
                                className="w-1 h-4 bg-red-400 rounded-full animate-pulse"
                                style={{ animationDelay: `${i * 0.2}s` }}
                            />
                        ))}
                    </div>
                )}

                <button
                    onClick={toggleListener}
                    className={`
            p-3 rounded-xl transition-all duration-300
            ${isListening
                            ? 'bg-red-500 text-white hover:bg-red-600'
                            : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-500/30'}
          `}
                >
                    {isListening ? <Mic size={20} /> : <MicOff size={20} />}
                </button>

                <div className="flex flex-col pr-2">
                    <span className="text-xs font-medium text-white/50 uppercase tracking-wider">
                        Voice Mode
                    </span>
                    <span className="text-sm font-semibold text-white">
                        {status === 'listening' ? 'Listening...' : status === 'processing' ? 'Processing...' : 'Off'}
                    </span>
                </div>
            </div>
        </div>
    );
};
