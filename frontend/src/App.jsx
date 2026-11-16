import { useState } from 'react'
import VideoLoader from './components/VideoLoader'
import ChatWindow from './components/ChatWindow'

export default function App() {
  // null = no video loaded, string = loaded video ID
  const [loadedVideoId, setLoadedVideoId] = useState(null)

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🎬</span>
        <div>
          <h1 className="text-xl font-bold text-white">YouTube RAG Chatbot</h1>
          <p className="text-xs text-gray-400">Ask questions about any YouTube video</p>
        </div>
        {loadedVideoId && (
          <div className="ml-auto flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-gray-400">
              Video loaded:{' '}
              <span className="text-green-400 font-mono">{loadedVideoId}</span>
            </span>
            <button
              onClick={() => setLoadedVideoId(null)}
              className="ml-2 text-xs text-gray-500 hover:text-red-400 transition-colors"
            >
              ✕ Change video
            </button>
          </div>
        )}
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center p-4">
        {!loadedVideoId ? (
          <VideoLoader onVideoLoaded={setLoadedVideoId} />
        ) : (
          <ChatWindow videoId={loadedVideoId} />
        )}
      </main>
    </div>
  )
}
