import { API_BASE_URL } from './connection'
import type { ChecklistResponse } from '../types/checklist'

export async function generateChecklist(productName: string, intendedUse: string): Promise<ChecklistResponse> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 180_000) // 3 minute timeout for LLM generation

  try {
    const response = await fetch(`${API_BASE_URL}/api/checklist/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product: productName, intended_use: intendedUse }),
      signal: controller.signal,
    });

    if (!response.ok) {
      // Extract the backend's detailed error message
      let detail = `Server error (${response.status})`
      try {
        const errorBody = await response.json()
        if (errorBody.detail) {
          detail = errorBody.detail
        }
      } catch {
        // If error body isn't JSON, use status text
        detail = `API Error: ${response.status} ${response.statusText}`
      }
      throw new Error(detail)
    }

    const data = await response.json()
    return data as ChecklistResponse
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Checklist generation timed out. The LLM may be overloaded — please try again.')
    }
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
      throw new Error('Cannot reach the backend server. Please ensure the server is running on port 8000.')
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}
