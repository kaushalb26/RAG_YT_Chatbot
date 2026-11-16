import { useState, useRef, useEffect } from 'react'

export default function ChatWindow({ videoId }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Video **${videoId}** is loaded! Ask me anything about it. 🎬`,
    },
  ])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleAsk = async () => {
    const trimmed = question.trim()
    if (!trimmed) return
    if (loading) return

    // Add user message to chat immediately
    const userMessage = { role: 'user', content: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.detail || 'Something went wrong. Please try again.')
        return
      }

      // Add AI response to chat
      const assistantMessage = { role: 'assistant', content: data.answer }
      setMessages((prev) => [...prev, assistantMessage])

    } catch (err) {
      setError('Could not connect to the server. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  return (
    <div className="w-full max-w-3xl flex flex-col h-[80vh]">
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 pb-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`flex items-start gap-3 max-w-[85%] ${
                msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              {/* Avatar */}
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300'
                }`}
              >
                {msg.role === 'user' ? 'U' : '🤖'}
              </div>

              {/* Bubble */}
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-tr-sm'
                    : 'bg-gray-800 text-gray-100 rounded-tl-sm'
                }`}
              >
                {msg.content}
              </div>
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-sm">
                🤖
              </div>
              <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex gap-1 items-center h-4">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-3 px-4 py-2 bg-red-900/40 border border-red-700 rounded-xl text-red-300 text-sm flex items-center gap-2">
          <span>⚠️</span>
          <span>{error}</span>
          <button
            onClick={() => setError('')}
            className="ml-auto text-red-400 hover:text-red-200"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input row */}
      <div className="flex gap-3 items-end">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the video... (Enter to send)"
          disabled={loading}
          rows={1}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 
                     text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 
                     resize-none text-sm transition-colors disabled:opacity-50"
        />
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed 
                     text-white px-5 py-3 rounded-xl transition-colors flex items-center gap-2 
                     font-semibold text-sm whitespace-nowrap"
        >
          {loading ? (
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <>Send ➤</>
          )}
        </button>
      </div>

      <p className="text-xs text-gray-600 text-center mt-2">
        Answers are based only on the video transcript · Press Enter to send
      </p>
    </div>
  )
}
