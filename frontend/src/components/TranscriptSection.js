"use client";

import { 
  FileText, 
  Download, 
  Search, 
  Copy, 
  ChevronDown, 
  ChevronRight, 
  List, 
  Filter, 
  Share2,
  Clock,
  X,
  MessageSquare,
  AlignJustify,
  BookOpen,
  Hash
} from "lucide-react";
import PropTypes from 'prop-types';
import { useState, useRef, useEffect, useMemo, useCallback } from 'react';

// Helper function to format time (e.g., 75.5 → 01:15)
const formatTime = (seconds) => {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

// Helper function to safely process topics
const processTopics = (topics) => {
  if (!topics || !Array.isArray(topics)) return [];
  return topics
    .map(topic => {
      if (typeof topic === 'string') return topic;
      if (typeof topic === 'object' && topic !== null) {
        return topic.name || topic.label || topic.title || 
               (topic.id ? `Topic ${topic.id}` : null);
      }
      return null;
    })
    .filter(Boolean);
};

// Segment processor to create sentence-level segments
const processSentenceSegments = (segments) => {
  if (!segments || !Array.isArray(segments)) return [];
  
  return segments.map(segment => ({
    ...segment,
    isPartial: false
  }));
};

const TranscriptSection = ({ 
  transcript = '',
  segments = [],
  topics = []
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCopied, setIsCopied] = useState(false);
  const [viewMode, setViewMode] = useState('sentences');
  const [activeFilters, setActiveFilters] = useState([]);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [fontSize, setFontSize] = useState('normal');
  const [copyError, setCopyError] = useState(null);
  const transcriptRef = useRef(null);
  const searchMatchRefs = useRef({});
  const searchTimeoutRef = useRef(null);
  
  // Process topics
  const processedTopics = useMemo(() => processTopics(topics), [topics]);
  
  // Handle different transcript formats
  const fullText = useMemo(() => {
    if (typeof transcript === 'string') return transcript;
    if (Array.isArray(transcript?.segments)) {
      return transcript.segments.map(s => s.text).join(' ');
    }
    return '';
  }, [transcript]);
    
  const originalSegments = useMemo(() => {
    if (Array.isArray(segments) && segments.length > 0) return segments;
    if (Array.isArray(transcript?.segments)) return transcript.segments;
    return [];
  }, [segments, transcript]);

  // Process segments by sentences
  const sentenceSegments = useMemo(() => 
    processSentenceSegments(originalSegments),
    [originalSegments]
  );

  // Get the appropriate segments based on view mode
  const processedSegments = useMemo(() => {
    const segments = viewMode === 'sentences' ? sentenceSegments : originalSegments;
    return segments.map((segment, index) => ({
      ...segment,
      uniqueId: `${viewMode}-${index}-${segment.start.toFixed(3)}-${segment.end.toFixed(3)}`
    }));
  }, [viewMode, sentenceSegments, originalSegments]);

  // Calculate transcript statistics
  const stats = useMemo(() => ({
    duration: processedSegments.length > 0 
      ? Math.round(processedSegments[processedSegments.length - 1].end)
      : 0,
    wordCount: fullText.split(/\s+/).length,
    segmentCount: processedSegments.length,
    topicCount: processedTopics.length
  }), [processedSegments, fullText, processedTopics]);

  // Filter segments based on search and active filters
  const filteredSegments = useMemo(() => {
    const searchLower = searchQuery.toLowerCase();
    return processedSegments.filter(segment => {
      const matchesSearch = !searchQuery || 
        segment.text.toLowerCase().includes(searchLower);
      const matchesTopics = activeFilters.length === 0 || 
        activeFilters.some(topic => 
          segment.text.toLowerCase().includes(topic.toLowerCase())
        );
      return matchesSearch && matchesTopics;
    });
  }, [processedSegments, searchQuery, activeFilters]);

  // Callbacks for user actions
  const toggleTopicFilter = useCallback((topic) => {
    setActiveFilters(prev => 
      prev.includes(topic)
        ? prev.filter(t => t !== topic)
        : [...prev, topic]
    );
  }, []);

  const handleFontSizeChange = useCallback((size) => {
    setFontSize(size);
  }, []);

  const shareSegment = useCallback(async (segment) => {
    try {
      const text = `${segment.text} [${formatTime(segment.start)}]`;
      await navigator.clipboard.writeText(text);
      setSelectedSegment(segment.uniqueId);
      setTimeout(() => setSelectedSegment(null), 2000);
    } catch (error) {
      console.error('Failed to copy segment:', error);
      setCopyError('Failed to copy segment to clipboard');
      setTimeout(() => setCopyError(null), 2000);
    }
  }, []);

  const toggleViewMode = useCallback(() => {
    setViewMode(prev => prev === 'sentences' ? 'segments' : 'sentences');
  }, []);
  
  const handleSearch = useCallback((e) => {
    setSearchQuery(e.target.value);
  }, []);
  
  const highlightSearchMatches = useCallback((text) => {
    if (!searchQuery || !text) return text;
    try {
      const escapedQuery = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const parts = text.split(new RegExp(`(${escapedQuery})`, 'gi'));
      return parts.map((part, i) => 
        part.toLowerCase() === searchQuery.toLowerCase() 
          ? <mark key={i} className="bg-yellow-600/30 text-white">{part}</mark> 
          : part
      );
    } catch (error) {
      console.error('Error highlighting search matches:', error);
      return text; // Return original text if there's an error
    }
  }, [searchQuery]);
  
  const copyToClipboard = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(fullText);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy transcript:', error);
      setCopyError('Failed to copy transcript to clipboard');
      setTimeout(() => setCopyError(null), 2000);
    }
  }, [fullText]);
  
  const downloadTranscript = useCallback(() => {
    const element = document.createElement("a");
    const file = new Blob([fullText], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = "transcript.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }, [fullText]);

  // Scroll to first match when search query changes
  useEffect(() => {
    if (!searchQuery || !processedSegments || processedSegments.length === 0) return;
    
    // Clear any existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    searchTimeoutRef.current = setTimeout(() => {
      const firstMatchIndex = processedSegments.findIndex(segment => 
        segment && segment.text && segment.text.toLowerCase().includes(searchQuery.toLowerCase())
      );
      
      if (firstMatchIndex >= 0 && searchMatchRefs.current && searchMatchRefs.current[firstMatchIndex]) {
        searchMatchRefs.current[firstMatchIndex].scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
      }
    }, 100);

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
        searchTimeoutRef.current = null;
      }
    };
  }, [searchQuery, processedSegments]);

  if (!fullText) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/50 border border-slate-700/40">
        <div className="flex items-center gap-3 text-slate-400">
          <FileText size={20} />
          <span>No transcript available</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/40 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-700/40">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <FileText className="text-blue-400" size={20} />
            <h2 className="text-lg font-medium text-slate-200">Transcript</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="Copy transcript"
            >
              {isCopied ? (
                <span className="text-green-400 text-sm">Copied!</span>
              ) : (
                <Copy size={16} className="text-slate-400" />
              )}
            </button>
            <button
              onClick={downloadTranscript}
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="Download transcript"
            >
              <Download size={16} className="text-slate-400" />
            </button>
            <button
              onClick={() => setShowTimestamps(!showTimestamps)}
              className={`p-2 rounded-lg transition-colors ${
                showTimestamps ? 'bg-slate-700/50 text-blue-400' : 'hover:bg-slate-700/50 text-slate-400'
              }`}
              title="Toggle timestamps"
            >
              <Clock size={16} />
            </button>
          </div>
        </div>

        {/* Search and Controls */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={handleSearch}
              placeholder="Search transcript..."
              className="w-full pl-10 pr-4 py-2 bg-slate-800/60 border border-slate-700/40 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-300"
              >
                <X size={14} />
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={toggleViewMode}
              className="px-3 py-2 bg-slate-800/60 border border-slate-700/40 rounded-lg text-slate-300 hover:bg-slate-700/50 transition-colors flex items-center gap-2"
            >
              {viewMode === 'sentences' ? (
                <>
                  <MessageSquare size={16} />
                  <span className="text-sm">Sentences</span>
                </>
              ) : (
                <>
                  <AlignJustify size={16} />
                  <span className="text-sm">Segments</span>
                </>
              )}
            </button>
            <div className="flex items-center gap-1 px-2 bg-slate-800/60 border border-slate-700/40 rounded-lg">
              <button
                onClick={() => handleFontSizeChange('small')}
                className={`p-2 rounded transition-colors ${
                  fontSize === 'small' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-300'
                }`}
                title="Small text"
              >
                <span className="text-xs">A</span>
              </button>
              <button
                onClick={() => handleFontSizeChange('normal')}
                className={`p-2 rounded transition-colors ${
                  fontSize === 'normal' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-300'
                }`}
                title="Normal text"
              >
                <span className="text-sm">A</span>
              </button>
              <button
                onClick={() => handleFontSizeChange('large')}
                className={`p-2 rounded transition-colors ${
                  fontSize === 'large' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-300'
                }`}
                title="Large text"
              >
                <span className="text-base">A</span>
              </button>
            </div>
          </div>
        </div>

        {/* Statistics */}
        <div className="flex flex-wrap gap-4 mt-4 text-sm">
          <div className="flex items-center gap-2 text-slate-400">
            <Clock size={14} />
            <span>{formatTime(stats.duration)}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <BookOpen size={14} />
            <span>{stats.wordCount} words</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <MessageSquare size={14} />
            <span>{stats.segmentCount} segments</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <Hash size={14} />
            <span>{stats.topicCount} topics</span>
          </div>
        </div>

        {/* Topic Filters */}
        {processedTopics.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            {processedTopics.map((topic, index) => (
              <button
                key={index}
                onClick={() => toggleTopicFilter(topic)}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                  activeFilters.includes(topic)
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'bg-slate-700/40 text-slate-300 border border-slate-600/30 hover:bg-slate-700/60'
                }`}
              >
                {topic}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Transcript Content */}
      <div 
        ref={transcriptRef}
        className={`relative max-h-[600px] overflow-y-auto ${
          fontSize === 'small' ? 'text-sm' : fontSize === 'large' ? 'text-lg' : 'text-base'
        }`}
      >
        {filteredSegments.length === 0 ? (
          <div className="p-6 text-center text-slate-400">
            <p>No matching segments found</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredSegments.map((segment) => (
              <div
                key={segment.uniqueId}
                className={`group relative p-4 rounded-lg transition-colors ${
                  selectedSegment === segment.uniqueId
                    ? 'bg-blue-500/20 border border-blue-500/30'
                    : 'bg-slate-800/40 border border-slate-700/40 hover:bg-slate-700/30'
                }`}
              >
                {/* Segment content */}
                <div className="flex items-start gap-3">
                  {showTimestamps && (
                    <div className="flex-shrink-0 text-sm text-slate-400 font-mono mt-1">
                      {formatTime(segment.start)}
                    </div>
                  )}
                  <div className="flex-1 text-slate-200">
                    {highlightSearchMatches(segment.text)}
                  </div>
                </div>
                
                {/* Segment actions */}
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => shareSegment(segment)}
                    className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-300 transition-colors"
                    title="Copy segment"
                  >
                    <Share2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Copy Error Message */}
      {copyError && (
        <div className="fixed bottom-4 right-4 bg-red-500/90 text-white px-4 py-2 rounded-lg shadow-lg">
          {copyError}
        </div>
      )}
    </div>
  );
};

TranscriptSection.propTypes = {
  transcript: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.shape({
      segments: PropTypes.arrayOf(PropTypes.shape({
        text: PropTypes.string,
        start: PropTypes.number,
        end: PropTypes.number
      }))
    })
  ]),
  segments: PropTypes.arrayOf(PropTypes.shape({
    text: PropTypes.string,
    start: PropTypes.number,
    end: PropTypes.number
  })),
  topics: PropTypes.arrayOf(PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.shape({
      name: PropTypes.string,
      label: PropTypes.string,
      title: PropTypes.string,
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
    })
  ]))
};

export default TranscriptSection; 