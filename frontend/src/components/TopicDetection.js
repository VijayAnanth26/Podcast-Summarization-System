"use client";

import React, { useState, useMemo } from "react";
import { BarChart2, AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import PropTypes from 'prop-types';

// Common topic name improvements
const improveTopicName = (name) => {
  if (!name) return "General Topic";
  
  // Clean up the topic name
  let improved = name.trim();
  
  // Handle common BERTopic naming patterns
  if (improved.match(/^topic_\d+$/i)) {
    return "General Topic";
  }
  
  // Remove numeric prefixes like "0_" or "Topic_12_"
  improved = improved.replace(/^\d+_|^topic_\d+_/i, '');
  
  // Capitalize each word
  improved = improved.split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
    
  // Handle single letter topics
  if (improved.length <= 1) {
    return "General Topic";
  }
  
  return improved;
};

const TopicDetection = ({ 
  topics = [],
  transcript = ""
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Process topics to ensure they have all required fields
  const processedTopics = useMemo(() => {
    // If no topics but we have transcript, create a simple topic
    if ((!topics || !Array.isArray(topics) || topics.length === 0) && transcript) {
      // Create a basic topic based on transcript content
      const words = transcript.split(/\s+/)
        .filter(word => word.length > 3)
        .filter(word => !['this', 'that', 'with', 'from', 'have', 'were', 'they'].includes(word.toLowerCase()));
      
      // Count word frequency
      const wordCount = {};
      words.forEach(word => {
        const cleanWord = word.toLowerCase().replace(/[^\w\s]/g, '');
        if (cleanWord.length > 3) {
          wordCount[cleanWord] = (wordCount[cleanWord] || 0) + 1;
        }
      });
      
      // Sort by frequency
      const topWords = Object.entries(wordCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([word]) => word);
      
      if (topWords.length > 0) {
        return [{
          name: `${topWords[0].charAt(0).toUpperCase() + topWords[0].slice(1)}`,
          keywords: topWords,
          score: 1,
          color: `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`
        }];
      }
      
      // Fallback if no good words found
      return [{
        name: "General Topic",
        keywords: [],
        score: 1,
        color: `hsl(210, 70%, 50%)`
      }];
    }

    if (!topics || !Array.isArray(topics) || topics.length === 0) {
      return [];
    }

    return topics.map(topic => {
      // Handle if topic is just a string
      if (typeof topic === 'string') {
        return {
          name: improveTopicName(topic),
          keywords: [],
          score: 0,
          color: `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`
        };
      }
      
      // Handle if topic is an object
      const processedTopic = {
        name: improveTopicName(topic.name || topic.label || topic.title || 'Unnamed Topic'),
        keywords: topic.keywords || topic.terms || topic.words || [],
        score: topic.score || topic.weight || topic.importance || 0,
        color: topic.color || `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`
      };
      
      // Ensure keywords is an array
      if (!Array.isArray(processedTopic.keywords)) {
        processedTopic.keywords = [];
      }
      
      return processedTopic;
    });
  }, [topics, transcript]);

  if (!processedTopics || processedTopics.length === 0) {
    return (
      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
        <div className="flex items-center mb-4">
          <BarChart2 className="mr-3 text-blue-400" size={18} />
          <h2 className="text-lg font-medium text-slate-200">Topics</h2>
        </div>
        <div className="text-slate-400 text-sm p-3 bg-slate-800/70 rounded-lg border border-slate-700/50">
          <div className="flex items-start">
            <AlertTriangle size={16} className="text-amber-400 mr-2 mt-0.5 flex-shrink-0" />
            <p className="text-amber-300">No topics were detected in this content.</p>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <BarChart2 className="mr-3 text-blue-400" size={18} />
          <h2 className="text-lg font-medium text-slate-200">Topics</h2>
        </div>
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-slate-400 hover:text-slate-300 transition-colors flex items-center"
        >
          {isExpanded ? (
            <>
              <span className="mr-1">Show less</span>
              <ChevronDown size={14} />
            </>
          ) : (
            <>
              <span className="mr-1">Show details</span>
              <ChevronRight size={14} />
            </>
          )}
        </button>
      </div>
      
      <div className="space-y-3">
        {processedTopics.map((topic, index) => (
          <div key={`topic-${index}`} className="p-3 bg-slate-800/40 rounded-lg border border-slate-700/30 hover:bg-slate-800/60 transition-colors">
            <div className="flex items-center gap-3">
              <div 
                className="w-3 h-3 rounded-full flex-shrink-0" 
                style={{ backgroundColor: topic.color }}
              />
              <h3 className="text-slate-200 font-medium">{topic.name}</h3>
              
              {topic.score > 0 && (
                <div className="ml-auto text-xs text-slate-400">
                  Score: {(topic.score * 100).toFixed(0)}%
                </div>
              )}
            </div>
            
            {isExpanded && topic.keywords && topic.keywords.length > 0 && (
              <div className="mt-2 pl-6">
                <div className="flex flex-wrap gap-1 mt-1">
                  {topic.keywords.slice(0, 5).map((keyword, kidx) => (
                    <span 
                      key={`kw-${index}-${kidx}`}
                      className="px-2 py-0.5 bg-slate-700/50 rounded text-xs text-slate-300"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

TopicDetection.propTypes = {
  topics: PropTypes.arrayOf(
    PropTypes.oneOfType([
      PropTypes.string,
      PropTypes.shape({
        name: PropTypes.string,
        keywords: PropTypes.arrayOf(PropTypes.string),
        score: PropTypes.number,
        color: PropTypes.string
      })
    ])
  ),
  transcript: PropTypes.string
};

export default TopicDetection; 