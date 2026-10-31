import { fetchApi } from './apiClient';
import { resourceService } from './resourceService';
import { ScalingRecommendation, CostOptimizationItem, SystemEnergySummary } from '../types/cloudscaler';

export const optimizationService = {
  // Returns aggregate energy and power optimization summary from Policy B engine
  async getEnergySummary(): Promise<SystemEnergySummary> {
    try {
      return await fetchApi<SystemEnergySummary>('/api/v1/resources/energy/summary');
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return {
        total_resources: 110,
        baseline_power_w: 17631.91,
        recommended_power_w: 17431.72,
        baseline_energy_wh: 1469.32,
        recommended_energy_wh: 1452.64,
        energy_savings_wh: 16.68,
        energy_savings_kwh: 0.0167,
        energy_savings_percentage: 1.14,
        evaluation_interval_sec: 300.0,
        power_model: 'Linear (100W Idle - 250W Max)',
        is_simulated: true,
      };
    }
  },

  // Returns scaling recommendations backed by Policy B backend endpoint (/analyze)
  async getScalingRecommendations(): Promise<ScalingRecommendation[]> {
    try {
      const resources = await resourceService.getResources();
      
      // Concurrently query backend Policy B analysis for each resource
      const analysisPromises = resources.map(async (r, index) => {
        const id = `rec-${index + 1}`;
        try {
          const analysis = await fetchApi<any>(`/api/v1/resources/${r.id}/analyze`, { method: 'POST' });

          const status = analysis.status;
          let priority: 'HIGH' | 'MEDIUM' | 'LOW' = 'MEDIUM';
          if (status === 'OVERLOADED') priority = 'HIGH';
          else if (status === 'UNDERUTILIZED') priority = 'LOW';

          let perfImpact: 'INCREASE' | 'MAINTAIN' | 'OPTIMIZE' = 'MAINTAIN';
          let costImpact: 'INCREASE' | 'DECREASE' | 'NEUTRAL' = 'NEUTRAL';
          if (analysis.action_type === 'SCALE_UP') {
            perfImpact = 'INCREASE';
            costImpact = 'INCREASE';
          } else if (analysis.action_type === 'SCALE_DOWN') {
            perfImpact = 'OPTIMIZE';
            costImpact = 'DECREASE';
          }

          let estCostChange = 0;
          if (analysis.estimated_impact?.estimated_monthly_savings) {
            estCostChange = -Math.round(analysis.estimated_impact.estimated_monthly_savings);
          } else if (analysis.action_type === 'SCALE_UP') {
            estCostChange = Math.round(r.monthlyCost * 0.25);
          }

          return {
            id,
            resourceId: String(r.id),
            resourceName: r.name,
            currentStatus: status,
            cpuUtilization: r.cpu,
            memoryUtilization: r.memory,
            recommendation: analysis.suggested_action || analysis.recommendation,
            reason: analysis.decision_reason || analysis.recommendation,
            priority,
            impact: {
              performance: perfImpact,
              availability: analysis.action_type === 'SCALE_UP' ? ('INCREASE' as const) : ('STABLE' as const),
              cost: costImpact,
            },
            estimatedCostChange: estCostChange,
            actionType: analysis.action_type,
            policyId: analysis.policy_id,
            safeguardStatus: analysis.safeguard_status,
            cooldownRemainingSteps: analysis.cooldown_remaining_steps,
            allocatedCapacity: analysis.allocated_capacity,
            costEstimate: analysis.cost_estimate,
            energyEstimate: analysis.energy_estimate,
            isSimulated: analysis.is_simulated ?? true,
          };
        } catch {
          // Fallback to local heuristic mapping if /analyze fails
          return {
            id,
            resourceId: String(r.id),
            resourceName: r.name,
            currentStatus: r.status,
            cpuUtilization: r.cpu,
            memoryUtilization: r.memory,
            recommendation: r.status === 'OVERLOADED'
              ? `Scale Up Instance Capacity (Increase from ${r.capacity} to ${r.capacity * 2} Units)`
              : r.status === 'UNDERUTILIZED'
              ? `Downsize Capacity / Right-Size VM (Decrease from ${r.capacity} Unit to Micro Tier)`
              : `Maintain Provisioned Capacity (${r.capacity} Units)`,
            reason: `Local fallback evaluation: CPU=${r.cpu}%, Mem=${r.memory}%.`,
            priority: r.status === 'OVERLOADED' ? ('HIGH' as const) : r.status === 'UNDERUTILIZED' ? ('LOW' as const) : ('MEDIUM' as const),
            impact: {
              performance: r.status === 'OVERLOADED' ? ('INCREASE' as const) : ('MAINTAIN' as const),
              availability: 'STABLE' as const,
              cost: r.status === 'OVERLOADED' ? ('INCREASE' as const) : r.status === 'UNDERUTILIZED' ? ('DECREASE' as const) : ('NEUTRAL' as const),
            },
            estimatedCostChange: r.status === 'UNDERUTILIZED' ? -Math.round(r.monthlyCost * 0.4) : r.status === 'OVERLOADED' ? Math.round(r.monthlyCost) : 0,
            policyId: 'Fallback Heuristic',
            isSimulated: true,
          };
        }
      });

      return await Promise.all(analysisPromises);
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return [];
    }
  },

  // Returns right-sizing and FinOps cost details dynamically calculated from live resource list
  async getCostOptimization(): Promise<{ items: CostOptimizationItem[]; totals: any }> {
    try {
      const resources = await resourceService.getResources();
      let totalCost = 0;
      let totalCapacity = 0;
      let potentialSavings = 0;

      const items = resources.map((r) => {
        totalCost += r.monthlyCost;
        totalCapacity += r.capacity;

        let recommendedSize = `Current (${r.capacity} Units)`;
        let estimatedCost = r.monthlyCost;
        let itemSavings = 0;
        let recommendation = 'Right-sized — no cost action needed';

        if (r.status === 'OVERLOADED') {
          recommendedSize = `Upgrade Capacity (${r.capacity * 2} Units)`;
          estimatedCost = Math.round(r.monthlyCost * 1.5);
          recommendation = 'Scaling required for performance reliability';
        } else if (r.status === 'UNDERUTILIZED') {
          recommendedSize = 'Downsized Tier (t3.micro)';
          estimatedCost = Math.round(r.monthlyCost * 0.6);
          itemSavings = Math.round(r.monthlyCost * 0.4);
          potentialSavings += itemSavings;
          recommendation = 'Right-size instance tier to eliminate idle capacity';
        }

        return {
          resourceId: r.id,
          resourceName: r.name,
          currentUtilization: { cpu: r.cpu, memory: r.memory },
          currentCost: r.monthlyCost,
          recommendedSize,
          estimatedCost,
          potentialSavings: itemSavings,
          recommendation,
          confidence: 'High' as const,
        };
      });

      return {
        items,
        totals: {
          totalResources: resources.length,
          totalMonthlyCost: totalCost,
          totalCapacity,
          potentialSavings,
          systemHealth: resources.some((r) => r.status === 'OVERLOADED')
            ? 'Action Required (High Load)'
            : 'Healthy',
        },
      };
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return {
        items: [],
        totals: {
          totalResources: 0,
          totalMonthlyCost: 0,
          totalCapacity: 0,
          potentialSavings: 0,
          systemHealth: 'Offline',
        },
      };
    }
  },
};
