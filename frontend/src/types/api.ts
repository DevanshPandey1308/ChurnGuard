export type FeatureValue = number | string

export interface CustomerPayload {
  CustomerID?: string | number | null
  snapshot_date?: string | null
  features: Record<string, FeatureValue>
}

export interface ScoreResult {
  CustomerID: string | number | null
  snapshot_date: string | null
  churn_probability: number
  predicted_90d_value: number
  risk_weighted_value: number
  churn_rank?: number | null
  value_rank?: number | null
  risk_value_rank?: number | null
}

export interface HealthResponse {
  status: 'ok' | 'not_ready' | string
  churn_model_loaded: boolean
  calibrator_loaded: boolean
  future_value_model_loaded: boolean
}

export interface BatchResponse {
  count: number
  scored_customers: ScoreResult[]
}
