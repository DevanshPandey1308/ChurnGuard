import { describe, expect, it } from 'vitest'
import importance from '../data/global_feature_importance.json'

describe('copied global SHAP project artifact', () => {
  it('preserves the exact top-ten feature ranking and source mean absolute magnitudes', () => {
    expect(importance).toHaveLength(10)
    expect(importance[0]).toEqual({ feature: 'purchase_invoice_count', mean_abs_shap: 0.44024064189410905, rank: 1 })
    expect(importance[9]).toEqual({ feature: 'average_items_per_order', mean_abs_shap: 0.09354742447622216, rank: 10 })
    expect(importance.map((row) => row.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  })
})
