export function resolveApiBaseUrl(value?: string): string {
  return (value?.trim() || 'http://localhost:8000').replace(/\/$/, '')
}
