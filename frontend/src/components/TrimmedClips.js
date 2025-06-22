"use client";

import React, { useState } from 'react';
import { X, Play, Pause, Download, Clock, FileText } from 'lucide-react';
import PropTypes from 'prop-types';

const formatTime = (seconds) => {
  if (!seconds && seconds !== 0) return "--:--";
  
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const TrimmedClips = ({ clips = [], onRemoveClip }) => {
  const [playingIndex, setPlayingIndex] = useState(null);
  
  // Handle play/pause toggle
  const togglePlay = (index) => {
    if (playingIndex === index) {
      setPlayingIndex(null);
    } else {
      setPlayingIndex(index);
    }
  };
  
  // If no clips, show empty state
  if (!clips || clips.length === 0) {
    return null;
  }
  
  return (
    <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
      <div className="flex items-center mb-4">
        <div className="flex items-center">
          <div className="w-6 h-6 rounded-md bg-blue-500/20 flex items-center justify-center mr-3">
            <FileText className="text-blue-400" size={16} />
          </div>
          <h2 className="text-lg font-medium text-slate-200">Trimmed Highlights</h2>
        </div>
      </div>
      
      <div className="space-y-4">
        {clips.map((clip, index) => (
          <div 
            key={clip.trim_id} 
            className="bg-slate-800/40 rounded-lg border border-slate-700/30 overflow-hidden"
          >
            <div className="p-3 border-b border-slate-700/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => togglePlay(index)}
                    className="w-8 h-8 rounded-full bg-blue-500 hover:bg-blue-600 flex items-center justify-center transition-colors"
                  >
                    {playingIndex === index ? (
                      <Pause size={16} className="text-white" />
                    ) : (
                      <Play size={16} className="text-white" />
                    )}
                  </button>
                  <div>
                    <p className="text-sm text-slate-300 font-medium">Highlight {index + 1}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <Clock size={12} />
                      <span>{formatTime(clip.start_time)} - {formatTime(clip.end_time)}</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <a 
                    href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${clip.audio_url}`}
                    download
                    className="p-1.5 rounded-md bg-slate-700/50 hover:bg-slate-700 transition-colors"
                    title="Download clip"
                  >
                    <Download size={16} className="text-slate-300" />
                  </a>
                  
                  {onRemoveClip && (
                    <button
                      onClick={() => onRemoveClip(clip.trim_id)}
                      className="p-1.5 rounded-md bg-red-500/10 hover:bg-red-500/20 transition-colors"
                      title="Remove clip"
                    >
                      <X size={16} className="text-red-400" />
                    </button>
                  )}
                </div>
              </div>
              
              {clip.highlight_text && (
                <div className="mt-2 text-sm text-slate-300 bg-slate-700/30 p-2 rounded">
                  <p>{clip.highlight_text}</p>
                </div>
              )}
            </div>
            
            {/* Audio/Video Player */}
            <div className={`w-full ${playingIndex === index ? 'block' : 'hidden'}`}>
              <audio
                src={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${clip.audio_url}`}
                className="w-full"
                controls
                autoPlay={playingIndex === index}
                onEnded={() => setPlayingIndex(null)}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

TrimmedClips.propTypes = {
  clips: PropTypes.arrayOf(
    PropTypes.shape({
      trim_id: PropTypes.string.isRequired,
      audio_url: PropTypes.string.isRequired,
      start_time: PropTypes.number.isRequired,
      end_time: PropTypes.number.isRequired,
      highlight_text: PropTypes.string,
    })
  ),
  onRemoveClip: PropTypes.func
};

export default TrimmedClips; 