# Train: zeta 9.0, 11.0 × alpha -1 to -5
for zeta in 9.0 11.0; do
  for alpha in 1.0 2.0 3.0 4.0 5.0; do
    hf download polymathic-ai/active_matter \
      data/train/active_matter_L_10.0_zeta_${zeta}_alpha_-${alpha}.hdf5 \
      --repo-type dataset --local-dir ./data
  done
done

# Valid: zeta 11.0
for alpha in 2.0 4.0; do
  hf download polymathic-ai/active_matter \
    data/valid/active_matter_L_10.0_zeta_11.0_alpha_-${alpha}.hdf5 \
    --repo-type dataset --local-dir ./data
done

# Test: zeta 11.0, alpha -2.0 和 -4.0
for alpha in 2.0 4.0; do
  hf download polymathic-ai/active_matter \
    data/test/active_matter_L_10.0_zeta_11.0_alpha_-${alpha}.hdf5 \
    --repo-type dataset --local-dir ./data
done