# Per-operation tolerances for dense batched ARA

The `kblas_{s,d,c,z}ara_batch_tol` entry points accept a device array of
`num_ops` tolerances instead of the scalar `tol` argument accepted by
`kblas_{s,d,c,z}ara_batch`. Each matrix uses its own threshold during the
existing adaptive randomized approximation (ARA) convergence test.

All other arguments retain their existing meanings. In particular:

- `relative=0`: the array contains absolute thresholds.
- `relative=1`: each threshold is multiplied by that matrix's maximum
  sampled diagonal magnitude, just as in the scalar API.
- The array uses `float` for single real/complex precision and `double`
  for double real/complex precision. Values must be finite and nonnegative;
  checking device array contents remains the caller's responsibility.
- The array must remain valid until operations on the handle's stream
  complete. Produce it on the same stream, or establish a dependency.
- `num_ops=0` succeeds without accessing the array. Negative counts and a
  null tolerance array for a nonempty batch return `KBLAS_Error_WrongInput`.

The original scalar APIs remain available and retain their convergence
behavior. The header defines `KBLAS_HAS_ARA_TOL_ARRAY` as `1` so applications
can conditionally use the extension when built with matching headers and
library binaries. No extra internal workspace is required.

For example, an application can compute a threshold for each tile `B`:

```
tau[B] = max(abs_tol, rel_tol * max_j ||B(:,j)||_2)
```

It can then pass the device array with `relative=0`. This particular scale
matches the first pivot magnitude of column-pivoted QR (without fixed
columns); KBLAS does not compute that norm itself. ARA still uses randomized
samples and its existing consecutive-small-vector stopping test, so this
does not imply identical ranks or error bounds to a QR-based compressor.

## Regression test

With SVD support enabled and the usual KBLAS test dependencies configured,
build `testing/bin/test_ara_tolerances_batch` through the test Makefile:

```
make -C testing bin/test_ara_tolerances_batch
./testing/bin/test_ara_tolerances_batch
```

The standalone test requires one GPU and no STRUMPACK dependency. It covers
all four scalar types, padded rectangular matrices, different thresholds
within one batch, approximation residuals, scalar/array rank equivalence
with the same random seed in both relative modes, and empty/invalid inputs.
