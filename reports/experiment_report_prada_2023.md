# Experiment Report

_Generated on 2025-10-01 22:47:12_

## Summary

| Experiment | Model | Tasks | Matches | Mismatches | Accuracy |
| --- | --- | --- | --- | --- | --- |
| PRADA S.P.A 2023 | google-generativegemini-25-flash | 18 | 18 | 0 | 100.0% |
| PRADA S.P.A 2023 | google-generativegemini-25-pro | 18 | 18 | 0 | 100.0% |
| PRADA S.P.A 2023 | groqgroqcompound-mini | 18 | 17 | 1 | 94.4% |
| PRADA S.P.A 2023 | groqopenaigpt-oss-120b | 18 | 16 | 2 | 88.9% |
| PRADA S.P.A 2023 | openai-compatiblegemma312b | 18 | 8 | 10 | 44.4% |
| PRADA S.P.A 2023 | openai-compatiblegemma34b | 18 | 11 | 7 | 61.1% |
| PRADA S.P.A 2023 | openai-compatiblellama32-visionlatest | 18 | 7 | 11 | 38.9% |
| PRADA S.P.A 2023 | openai-compatiblellama323b | 18 | 3 | 15 | 16.7% |
| PRADA S.P.A 2023 | openai-compatiblemetallama-32-11b-vision-instruct_nvidia_nim | 18 | 7 | 11 | 38.9% |
| PRADA S.P.A 2023 | openai-compatiblemetallama-32-90b-vision-instruct_nvidia_nim | 18 | 10 | 8 | 55.6% |
| PRADA S.P.A 2023 | openai-compatiblemetallama-4-maverick-17b-128e-instruct_nvidia_nim | 18 | 17 | 1 | 94.4% |
| PRADA S.P.A 2023 | openai-compatibleqwen38b | 18 | 5 | 13 | 27.8% |
| PRADA S.P.A 2023 | openai-compatibleqwenqwen3-235b-a22b_nvidia_nim | 18 | 18 | 0 | 100.0% |
| PRADA S.P.A 2023 | openai-compatibleqwenqwen3-next-80b-a3b-instruct_nvidia_nim | 18 | 16 | 2 | 88.9% |

## PRADA S.P.A — 2023

- **Industry**: fashion
- **Source document**: `data/NFD/PRADA S.P.A/NFD2023.pdf`
- **Metadata**: language: en; notes: Official sustainability report PDF in Italian

### Model: google-generativegemini-25-flash

- **Run folder**: `artifacts/prada_spa_2023_google-generativegemini-25-flash`
- **Performance**: Tasks: 18, Matches: 18, Mismatches: 0, Not found outputs: 8
- **Accuracy**: 100.0%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 kWh | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100% | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5199 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | not found | not found | ✅ yes |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: google-generativegemini-25-pro

- **Run folder**: `artifacts/prada_spa_2023_google-generativegemini-25-pro`
- **Performance**: Tasks: 18, Matches: 18, Mismatches: 0, Not found outputs: 8
- **Accuracy**: 100.0%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100% | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5,199 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | not found | not found | ✅ yes |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: groqgroqcompound-mini

- **Run folder**: `artifacts/prada_spa_2023_groqgroqcompound-mini`
- **Performance**: Tasks: 18, Matches: 17, Mismatches: 1, Not found outputs: 7
- **Accuracy**: 94.4%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3,976 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 tons | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 625 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3,513,368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384,218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 99 | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5162 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 312,568 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446,607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 28,364 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | higher | 5.135937 MWp | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | not found | not found | ✅ yes |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: groqopenaigpt-oss-120b

- **Run folder**: `artifacts/prada_spa_2023_groqopenaigpt-oss-120b`
- **Performance**: Tasks: 18, Matches: 16, Mismatches: 2, Not found outputs: 8
- **Accuracy**: 88.9%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3,976 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 tons | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | equal | 500 | ❌ no |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384,218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100% | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5,162 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306,548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446,607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | higher | 28364 | ❌ no |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | not found | not found | ✅ yes |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: openai-compatiblegemma312b

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblegemma312b`
- **Performance**: Tasks: 18, Matches: 8, Mismatches: 10, Not found outputs: 3
- **Accuracy**: 44.4%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | higher | 3646 | ❌ no |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 5135937 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | higher | 384218 | ❌ no |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 95 | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | 5199 | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | higher | {325992} | ❌ no |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | higher | 418748 | ❌ no |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | lower | not found | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | equal | not found | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal | not found | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower | 21000 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | lower | 5199 | ❌ no |

### Model: openai-compatiblegemma34b

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblegemma34b`
- **Performance**: Tasks: 18, Matches: 11, Mismatches: 7, Not found outputs: 4
- **Accuracy**: 61.1%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | {3,646} | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | {326 tons} | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | higher | {500} | ❌ no |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 5135937 kWh | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | {384,218} | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | {67%}, | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | {5199} | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306,548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | {446607} | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | equal | {11,578} | ❌ no |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | equal | 500000 | ❌ no |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | {not found} | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | {not found} | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | higher | {5,136} | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | {not found} | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | {not found} | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | higher | 78000 m3 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | higher | {5199} | ❌ no |

### Model: openai-compatiblellama32-visionlatest

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblellama32-visionlatest`
- **Performance**: Tasks: 18, Matches: 7, Mismatches: 11, Not found outputs: 0
- **Accuracy**: 38.9%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | ** 3,646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 tons. | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | equal | ** 500 | ❌ no |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher |  | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower |  | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher |  | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | {extracted_value or not found} | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | higher | 485,380 | ❌ no |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | higher | 500000 | ❌ no |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | equal | ** not found | ❌ no |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | lower | 111 | ❌ no |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | lower |  | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | equal | not found | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal |  | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | higher | 78% | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | lower | 22% | ❌ no |

### Model: openai-compatiblellama323b

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblellama323b`
- **Performance**: Tasks: 18, Matches: 3, Mismatches: 15, Not found outputs: 1
- **Accuracy**: 16.7%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | higher | 3,646 | ❌ no |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | lower | 300 | ❌ no |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | higher | 500 | ❌ no |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 5135937 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | higher | 384,218 | ❌ no |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | equal | 99 | ❌ no |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | 3,976 | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | equal | 306,548 | ❌ no |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | equal | not found | ❌ no |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | lower |  | ❌ no |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | higher | 100000 | ❌ no |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | higher | 5 | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | lower | not found | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal |  | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | higher | 21,000 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | equal | not found | ❌ no |

### Model: openai-compatiblemetallama-32-11b-vision-instruct_nvidia_nim

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblemetallama-32-11b-vision-instruct_nvidia_nim`
- **Performance**: Tasks: 18, Matches: 7, Mismatches: 11, Not found outputs: 1
- **Accuracy**: 38.9%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | higher | 5,199 | ❌ no |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher |  | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3,513,368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | higher |  | ❌ no |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | lower |  | ❌ no |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5,199 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | higher |  | ❌ no |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | equal |  | ❌ no |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | equal |  | ❌ no |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | equal |  | ❌ no |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | higher |  | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | equal | not found | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal |  | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower |  | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found |  | ✅ yes |

### Model: openai-compatiblemetallama-32-90b-vision-instruct_nvidia_nim

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblemetallama-32-90b-vision-instruct_nvidia_nim`
- **Performance**: Tasks: 18, Matches: 10, Mismatches: 8, Not found outputs: 1
- **Accuracy**: 55.6%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower |  | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher |  | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower |  | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 5,135,937 kWh | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower |  | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | lower |  | ❌ no |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower |  | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower |  | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | equal |  | ❌ no |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found |  | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | equal |  | ❌ no |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | higher | 5,136 MWh. | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | equal |  | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal |  | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower |  | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | equal |  | ❌ no |

### Model: openai-compatiblemetallama-4-maverick-17b-128e-instruct_nvidia_nim

- **Run folder**: `artifacts/prada_spa_2023_openai-compatiblemetallama-4-maverick-17b-128e-instruct_nvidia_nim`
- **Performance**: Tasks: 18, Matches: 17, Mismatches: 1, Not found outputs: 7
- **Accuracy**: 94.4%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100 | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5199 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower | 21 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: openai-compatibleqwen38b

- **Run folder**: `artifacts/prada_spa_2023_openai-compatibleqwen38b`
- **Performance**: Tasks: 18, Matches: 5, Mismatches: 13, Not found outputs: 0
- **Accuracy**: 27.8%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | equal | 500 | ❌ no |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | higher | 436897 | ❌ no |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | not found | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | 10643 | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 312568 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | higher | 446607 | ❌ no |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | higher | 28364 | ❌ no |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | higher | not found | ❌ no |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | equal | not found | ❌ no |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | higher | not found | ❌ no |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | equal | not found | ❌ no |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | higher | not found | ❌ no |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | equal | not found | ❌ no |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower | 21 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | higher | not found | ❌ no |

### Model: openai-compatibleqwenqwen3-235b-a22b_nvidia_nim

- **Run folder**: `artifacts/prada_spa_2023_openai-compatibleqwenqwen3-235b-a22b_nvidia_nim`
- **Performance**: Tasks: 18, Matches: 18, Mismatches: 0, Not found outputs: 8
- **Accuracy**: 100.0%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3,646 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 tons | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3,513,368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100% | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | lower | 5199 | ✅ yes |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 306548 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | not found | not found | ✅ yes |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |

### Model: openai-compatibleqwenqwen3-next-80b-a3b-instruct_nvidia_nim

- **Run folder**: `artifacts/prada_spa_2023_openai-compatibleqwenqwen3-next-80b-a3b-instruct_nvidia_nim`
- **Performance**: Tasks: 18, Matches: 16, Mismatches: 2, Not found outputs: 7
- **Accuracy**: 88.9%
- **Expected not found**: 8 task(s)

| Task | Indicator | Page | Benchmark | Expected | Model Output | Extracted Value | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prada_2023_total_purchased_raw_materials_p169 | total purchased parts | 169 | 4000 tons | lower | lower | 3,976 | ✅ yes |
| prada_2023_recycled_plastic_used_p90 | recycled plastic used | 90 | 300 tons | higher | higher | 326 tons | ✅ yes |
| prada_2023_total_plastic_used_p169 | total plastic used | 169 | 500 tons | lower | lower | 485 | ✅ yes |
| prada_2023_energy_produced_res_self_consumption_p84 | energy produced from RES for self-consumption | 84 | 3000000 kWh | higher | higher | 3513368 | ✅ yes |
| prada_2023_total_energy_consumed_p170 | total energy consumed | 170 | 400000 GJ | lower | lower | 384,218 | ✅ yes |
| prada_2023_recyclable_packaging_p99 | recyclable packaging | 99 | 95% | higher | higher | 100 | ✅ yes |
| prada_2023_total_packaging_p169 | total packaging | 169 | 5500 tons | lower | higher | 5162 | ❌ no |
| prada_2023_ghg_emissions_p171 | GHG emissions | 171 | 320000 tons of CO2e | lower | lower | 312,568 | ✅ yes |
| prada_2023_baseline_emissions_p163 | baseline emissions | 163 | 450000 tons of CO2e | lower | lower | 446607 | ✅ yes |
| prada_2023_current_emissions_p170 | current emissions | 170 | 15000 tons of CO2e | lower | lower | 11578 | ✅ yes |
| prada_2023_num_regenerated_components_p105 | number of regenerated components | 105 | 500000 components | not found | not found | not found | ✅ yes |
| prada_2023_weight_upcycled_materials_p106 | weight of upcycled materials | 106 | 5 tons | not found | not found | not found | ✅ yes |
| prada_2023_num_repaired_components_p111 | number of repaired components | 111 | 100000 repairs | not found | not found | not found | ✅ yes |
| prada_2023_power_res_plants_p84 | power of RES plants | 84 | 5 MWp | not found | not found | not found | ✅ yes |
| prada_2023_co2_captured_p75 | CO2 captured | 75 | 100 tons | not found | not found | not found | ✅ yes |
| prada_2023_co2_reused_p75 | CO2 reused | 75 | 10 tons | not found | not found | not found | ✅ yes |
| prada_2023_recovered_water_p102 | recovered water | 102 | 50.0k m3 | not found | lower | 21,000 | ❌ no |
| prada_2023_compostable_packaging_p98 | compostable packaging | 98 | 50 tons | not found | not found | not found | ✅ yes |
