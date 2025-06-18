"use client";
import { useState, useRef, useCallback, lazy, Suspense, useEffect } from "react";
import { UploadCloud, Youtube, Loader2, FileAudio, Info, ChevronRight, ChevronDown, AlertTriangle, Clock, X, FileText, BarChart2 } from "lucide-react";
import { useProcessing } from "@/hooks/useProcessing";
import { Podcast } from "lucide-react";

// Dynamically import components
const MetadataSection = lazy(() => import("@/components/MetadataSection"));
const TranscriptSection = lazy(() => import("@/components/TranscriptSection"));
const AbstractiveSummary = lazy(() => import("@/components/AbstractiveSummary"));
const ExtractiveSummary = lazy(() => import("@/components/ExtractiveSummary"));
const TopicDetection = lazy(() => import("@/components/TopicDetection"));

// Loading fallback component
const ComponentLoader = () => (
  <div className="animate-pulse bg-slate-800/60 rounded-xl p-6 w-full h-32">
    <div className="flex items-center">
      <div className="h-8 w-8 bg-slate-700/80 rounded-md mr-3"></div>
      <div className="h-6 bg-slate-700/80 rounded w-1/3"></div>
    </div>
    <div className="mt-4 space-y-3">
      <div className="h-4 bg-slate-700/60 rounded w-full"></div>
      <div className="h-4 bg-slate-700/60 rounded w-5/6"></div>
      <div className="h-4 bg-slate-700/60 rounded w-4/6"></div>
    </div>
  </div>
);

// YouTube URL validation regex
const YOUTUBE_URL_PATTERN = /^(https?:\/\/)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)\/(?:watch\?v=|shorts\/|embed\/|v\/|.+\?v=)?([a-zA-Z0-9_-]{11})(?:[&?].*)?$/;

// Function to validate YouTube URL
const isValidYouTubeURL = (url) => {
  if (!url || typeof url !== 'string') return false;
  return YOUTUBE_URL_PATTERN.test(url.trim());
};

// Function to format file size
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
};

export default function PodcastSummarizer() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [youtubeURL, setYoutubeURL] = useState("");
  const [activeTab, setActiveTab] = useState("upload");
  const [isHowItWorksOpen, setIsHowItWorksOpen] = useState(false);
  const fileInputRef = useRef(null);

  const {
    processing,
    progress,
    result,
    error,
    isLoadingResult,
    handleProcessing,
    setError,
    jobId
  } = useProcessing();

  // Reset states on fresh render
  useEffect(() => {
    setSelectedFile(null);
    setYoutubeURL("");
  }, []);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  // Reset state when switching tabs
  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab);
    setError(null);
    setSelectedFile(null);
    setYoutubeURL("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [setError]);

  const handleFileUpload = useCallback(async () => {
    if (!selectedFile) {
      setError("⚠ Please select a file to upload.");
      return;
    }

    // Validate file type
    const validTypes = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/ogg', 
                        'video/mp4', 'video/webm', 'video/ogg'];
    if (!validTypes.includes(selectedFile.type)) {
      setError("⚠ Invalid file type. Please upload an audio or video file.");
      return;
    }

    // Check file size (limit to 45MB to stay safely under 50MB limit)
    const MAX_FILE_SIZE = 45 * 1024 * 1024; // 45MB

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`⚠ File size exceeds the limit (45MB). Please upload a smaller file. Current size: ${formatFileSize(selectedFile.size)}`);
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    
    try {
      await handleProcessing("upload", formData, { "Content-Type": "multipart/form-data" });
    } catch (err) {
      // Additional error handling if needed
      if (err.message?.includes("timeout") || err.message?.includes("cancelled")) {
        setError("⚠ Request was cancelled - processing took too long. Try with a shorter audio file.");
      } else if (err.message?.includes("size") || err.message?.includes("large")) {
        setError(`⚠ File is too large (${formatFileSize(selectedFile.size)}). Please try a smaller file or compress it.`);
      }
    }
  }, [selectedFile, handleProcessing, setError]);

  const processYouTube = useCallback(async () => {
    if (!youtubeURL.trim()) {
      setError("⚠ Enter a valid YouTube URL.");
      return;
    }

    if (!isValidYouTubeURL(youtubeURL)) {
      setError("⚠ Invalid YouTube URL. Please provide a valid link.");
      return;
    }

    const formData = new FormData();
    formData.append("url", youtubeURL.trim());
    
    try {
      await handleProcessing("youtube", formData, { "Content-Type": "multipart/form-data" });
    } catch (err) {
      // Handle specific YouTube errors
      if (err.message?.includes("Copyright") || err.message?.includes("copyright")) {
        setError("⚠ This video has copyright restrictions and cannot be processed.");
      } else if (err.message?.includes("unavailable") || err.message?.includes("private")) {
        setError("⚠ This video is private or unavailable. Please try another video.");
      } else if (err.message?.includes("timeout") || err.message?.includes("cancelled")) {
        setError("⚠ Processing took too long and was cancelled. Try a shorter video or specify timestamps in URL.");
      }
    }
  }, [youtubeURL, handleProcessing, setError]);

  return (
    <main className="min-h-screen bg-slate-900 text-slate-200">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Left Sidebar - Input Card */}
          <div className="lg:w-[400px] flex-shrink-0">
            <div className="sticky top-8">
              <div className="bg-slate-800/50 rounded-xl border border-slate-700/40 overflow-hidden">
                {/* Header */}
                <div className="p-4 border-b border-slate-700/40">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                      <Podcast size={24} className="text-white" />
                    </div>
                    <h1 className="text-xl font-semibold text-slate-200">Podcast Summarizer</h1>
                  </div>
                  <p className="text-sm text-slate-400">
                    Transform your audio content into actionable insights with AI-powered transcription and summarization
                  </p>
                </div>

                {/* How it Works */}
                <div className="p-4 border-b border-slate-700/40">
                  <button
                    onClick={() => setIsHowItWorksOpen(!isHowItWorksOpen)}
                    className="w-full flex items-center justify-between text-left text-slate-200 hover:text-white transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Info size={20} />
                      <span className="font-medium">How it Works</span>
                    </div>
                    {isHowItWorksOpen ? (
                      <ChevronDown size={20} />
                    ) : (
                      <ChevronRight size={20} />
                    )}
                  </button>
                  {isHowItWorksOpen && (
                    <div className="mt-4 text-sm text-slate-300 space-y-2">
                      <p>1. Upload your podcast audio file or paste a YouTube URL</p>
                      <p>2. Our AI system will transcribe the audio content</p>
                      <p>3. Get detailed transcription, summaries, and topic analysis</p>
                      <p>4. Review and export your results</p>
                    </div>
                  )}
                </div>

                {/* Input Section */}
                <div className="p-4">
                  {/* Tabs */}
                  <div className="bg-slate-800/40 rounded-lg p-1 mb-4">
                    <div className="flex space-x-1">
                      <button
                        onClick={() => handleTabChange("upload")}
                        className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                          activeTab === "upload"
                            ? "bg-slate-700 text-white"
                            : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                        }`}
                      >
                        Upload Audio
                      </button>
                      <button
                        onClick={() => handleTabChange("youtube")}
                        className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                          activeTab === "youtube"
                            ? "bg-slate-700 text-white"
                            : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                        }`}
                      >
                        YouTube URL
                      </button>
                    </div>
                  </div>

                  {/* Upload Section */}
                  {activeTab === "upload" ? (
                    <div className="space-y-4">
                      <div className="w-full">
                        <label
                          htmlFor="file-upload"
                          className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
                        >
                          <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <UploadCloud className="w-8 h-8 mb-3 text-slate-400" />
                            <p className="text-sm text-slate-400">
                              <span className="font-semibold">Click to upload</span> or drag and drop
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              MP3, WAV, MP4 (max. 45MB)
                            </p>
                          </div>
                          <input
                            id="file-upload"
                            ref={fileInputRef}
                            type="file"
                            className="hidden"
                            onChange={handleFileChange}
                            accept="audio/*,video/*"
                          />
                        </label>
                      </div>
                      {selectedFile && (
                        <div className="flex items-center gap-2 text-sm text-slate-300">
                          <FileAudio size={16} />
                          <span>{selectedFile.name}</span>
                          <span className="text-slate-500">
                            ({formatFileSize(selectedFile.size)})
                          </span>
                          <button
                            onClick={() => {
                              setSelectedFile(null);
                              if (fileInputRef.current) {
                                fileInputRef.current.value = "";
                              }
                            }}
                            className="p-1 hover:bg-slate-700/50 rounded-full transition-colors"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      )}
                      <button
                        onClick={handleFileUpload}
                        disabled={processing || !selectedFile}
                        className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
                          processing || !selectedFile
                            ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                            : "bg-blue-600 hover:bg-blue-700 text-white"
                        }`}
                      >
                        {processing ? (
                          <div className="flex items-center justify-center gap-2">
                            <Loader2 className="animate-spin" size={16} />
                            <span>Processing...</span>
                          </div>
                        ) : (
                          "Process Audio"
                        )}
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-full space-y-2">
                        <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                          <Youtube size={16} />
                          <span>Enter YouTube URL</span>
                        </div>
                        <input
                          type="text"
                          value={youtubeURL}
                          onChange={(e) => setYoutubeURL(e.target.value)}
                          placeholder="https://www.youtube.com/watch?v=..."
                          className="w-full px-4 py-2 bg-slate-800/60 border border-slate-700/40 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                        />
                      </div>
                      <button
                        onClick={processYouTube}
                        disabled={processing || !youtubeURL.trim()}
                        className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
                          processing || !youtubeURL.trim()
                            ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                            : "bg-blue-600 hover:bg-blue-700 text-white"
                        }`}
                      >
                        {processing ? (
                          <div className="flex items-center justify-center gap-2">
                            <Loader2 className="animate-spin" size={16} />
                            <span>Processing...</span>
                          </div>
                        ) : (
                          "Process Video"
                        )}
                      </button>
                    </div>
                  )}

                  {/* Error Message */}
                  {error && (
                    <div className="mt-4 flex items-center gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                      <AlertTriangle size={16} />
                      <span>{error}</span>
                    </div>
                  )}

                  {/* Progress Bar */}
                  {processing && progress > 0 && (
                    <div className="mt-4 space-y-2">
                      <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-300 ease-in-out"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                      <p className="text-sm text-slate-400 text-center">
                        Processing: {Math.round(progress)}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Main Content Area - Results */}
          <div className="flex-1 space-y-6">
            {/* Rest of the content - Only show when result is available */}
            {(result || isLoadingResult) && (
              <Suspense fallback={<ComponentLoader />}>
                {result?.metadata && (
                  <MetadataSection metadata={result.metadata} />
                )}

                {result?.topics && (
                  <TopicDetection 
                    topics={result.topics} 
                    transcript={result.transcript}
                  />
                )}

                {result?.transcript && (
                  <TranscriptSection
                    transcript={result.transcript}
                    segments={result.segments}
                    topics={result.topics}
                  />
                )}

                {/* Summaries */}
                {result && (
                  <>
                    <Suspense fallback={<ComponentLoader />}>
                      <AbstractiveSummary 
                        summary={result.abstractiveSummary || result.abstractive_summary} 
                        transcript={result.transcript}
                      />
                    </Suspense>

                    <Suspense fallback={<ComponentLoader />}>
                      <ExtractiveSummary 
                        summary={result.extractiveSummary || result.extractive_summary}
                        transcript={result.transcript}
                      />
                    </Suspense>
                  </>
                )}
              </Suspense>
            )}

            {/* Initial State */}
            {!result && !isLoadingResult && !processing && (
              <div className="text-center py-12">
                <p className="text-slate-400">
                  Upload an audio file or provide a YouTube URL to get started
                </p>
              </div>
            )}

            {/* Processing State */}
            {processing && !result && (
              <div className="p-6 rounded-xl bg-slate-800/50 border border-slate-700/40">
                <div className="flex items-center gap-3 mb-4">
                  <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                  <h3 className="text-lg font-medium text-slate-200">
                    Processing Your Content
                  </h3>
                </div>
                <p className="text-slate-300 mb-4">
                  Please wait while we process your content. This may take a few minutes.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}