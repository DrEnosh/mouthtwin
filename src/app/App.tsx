import { Component, type ReactNode } from 'react'
import { Landing } from '../features/landing/Landing'
import { ProcessingView } from '../features/processing/ProcessingView'
import { Workspace } from '../features/workspace/Workspace'
import { useMouthTwin } from '../store/useMouthTwin'

class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div className="grid h-full place-items-center p-6 text-center">
        <div className="max-w-sm">
          <p className="text-[15px] text-ink">Something went wrong while rendering.</p>
          <p className="mt-2 text-[13px] text-muted">
            Your browser may not support WebGL, or the image could not be processed. Reload to start again.
          </p>
          <button
            onClick={() => location.reload()}
            className="mt-4 rounded-md border border-line-strong px-4 py-2 text-[13px] text-ink hover:border-accent/60"
          >
            Reload
          </button>
        </div>
      </div>
    )
  }
}

export function App() {
  const stage = useMouthTwin((s) => s.stage)
  return (
    <ErrorBoundary>
      {stage === 'landing' && <Landing />}
      {stage === 'processing' && <ProcessingView />}
      {stage === 'workspace' && <Workspace />}
    </ErrorBoundary>
  )
}
