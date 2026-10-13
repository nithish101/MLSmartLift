export function Loading({ children }) {
    return (
        <div className="status" role="status">
            <span className="status-pulse" aria-hidden="true" />
            {children}
        </div>
    );
}

export function LoadError() {
    return (
        <div className="status status--error" role="alert">
            Couldn&apos;t reach the SmartLift API. Check that the backend is running on port 8000.
        </div>
    );
}
