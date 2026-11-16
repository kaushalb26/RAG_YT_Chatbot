import { useState } from 'react'

export default function VideoLoader({ onVideoLoaded }) {
  const [videoId, setVideoId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleLoad = async () => {
    const trimmed = videoId.trim()
    if (!trimmed) {
      setError('Please enter a YouTube Video ID.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/load-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: trimmed }),
      })

      const data = await res.json()

      if (!res.ok) {
        // FastAPI sends error details in data.detail
        setError(data.detail || 'Failed to load video. Please try again.')
        return
      }

      // Success — notify parent App
      onVideoLoaded(data.video_id)

    } catch (err) {
      setError('Could not connect to the server. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleLoad()
  }

  return (
    <div className="w-full max-w-md">
      {/* Card */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl">
        <div className="text-center mb-8">
          <div className="text-5xl mb-4">🎬</div>
          <h2 className="text-2xl font-bold text-white mb-2">Load a YouTube Video</h2>
          <p className="text-gray-400 text-sm">
            Enter a YouTube Video ID to fetch its transcript and start chatting.
          </p>
        </div>

        {/* Example hint */}
        <p className="text-xs text-gray-500 mb-2">
          Example: for{' '}
          <span className="font-mono text-gray-400">
            youtube.com/watch?v=<span className="text-yellow-400">Gfr50f6ZBvo</span>
          </span>
          , enter{' '}
          <span className="font-mono text-yellow-400">Gfr50f6ZBvo</span>
        </p>

        {/* Input */}
        <input
          type="text"
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. Gfr50f6ZBvo"
          disabled={loading}
          className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white 
                     placeholder-gray-500 focus:outline-none focus:border-blue-500 
                     font-mono text-sm transition-colors disabled:opacity-50"
        />

        {/* Error message */}
        {error && (
          <p className="mt-3 text-red-400 text-sm flex items-start gap-2">
            <span>⚠️</span>
            <span>{error}</span>
          </p>
        )}

        {/* Load button */}
        <button
          onClick={handleLoad}
          disabled={loading}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 
                     disabled:cursor-not-allowed text-white font-semibold py-3 px-6 
                     rounded-xl transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Fetching transcript & building index...
            </>
          ) : (
            <>
              <span>🚀</span> Load Video
            </>
          )}
        </button>

        {/* Info note */}
        <p className="mt-4 text-xs text-gray-600 text-center">
          ⏱ First load may take ~30–60 seconds to embed the transcript.
        </p>
      </div>
    </div>
  )
}
