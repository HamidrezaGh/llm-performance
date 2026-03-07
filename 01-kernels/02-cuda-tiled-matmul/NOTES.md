# CUDA Tiled Matrix Multiplication Explained

This note explains the tiled matrix multiplication approach used in CUDA with a concrete numeric example.

The goal is to understand how the kernel computes:

```text
C = A x B
```

and how tiling + shared memory improves performance.

## 1. Example Matrices

Let:

A:
```text
1  2  3  4
5  6  7  8
9 10 11 12
13 14 15 16
```

B:
```text
1 0 2 1
3 1 0 2
4 1 1 0
2 3 0 1
```

Matrix sizes:
- A: 4 x 4
- B: 4 x 4
- C: 4 x 4

## 2. Tile Size

Assume:

```text
TILE_SIZE = 2
```

So matrices are divided into 2x2 tiles.

## 3. Tiles of Matrix A

```text
A00 | A01
----+----
A10 | A11
```

Tiles:

A00:
```text
1 2
5 6
```

A01:
```text
3 4
7 8
```

A10:
```text
9 10
13 14
```

A11:
```text
11 12
15 16
```

## 4. Tiles of Matrix B

```text
B00 | B01
----+----
B10 | B11
```

Tiles:

B00:
```text
1 0
3 1
```

B01:
```text
2 1
0 2
```

B10:
```text
4 1
2 3
```

B11:
```text
1 0
0 1
```

## 5. Computing One Output Tile

To compute tile `C00`, we multiply and accumulate:

```text
C00 = A00 x B00 + A01 x B10
```

This corresponds to the CUDA loop:

```cpp
for (int t = 0; t < numTiles; t++)
```

Each iteration processes one pair of tiles.

## 6. Phase `t = 0`

Load tiles into shared memory:

```text
sA = A00
sB = B00
```

sA:
```text
1 2
5 6
```

sB:
```text
1 0
3 1
```

Each thread computes a partial value.

### Thread (0,0) -> `C[0][0]`

```text
1x1 + 2x3
= 1 + 6
= 7
```

### Thread (0,1) -> `C[0][1]`

```text
1x0 + 2x1
= 2
```

### Thread (1,0) -> `C[1][0]`

```text
5x1 + 6x3
= 5 + 18
= 23
```

### Thread (1,1) -> `C[1][1]`

```text
5x0 + 6x1
= 6
```

Partial result after phase 0:

```text
7   2
23  6
```

## 7. Phase `t = 1`

Load next tiles:

```text
sA = A01
sB = B10
```

sA:
```text
3 4
7 8
```

sB:
```text
4 1
2 3
```

Compute again.

### Thread (0,0)

```text
3x4 + 4x2
= 12 + 8
= 20

Total:
7 + 20 = 27
```

### Thread (0,1)

```text
3x1 + 4x3
= 3 + 12
= 15

Total:
2 + 15 = 17
```

### Thread (1,0)

```text
7x4 + 8x2
= 28 + 16
= 44

Total:
23 + 44 = 67
```

### Thread (1,1)

```text
7x1 + 8x3
= 7 + 24
= 31

Total:
6 + 31 = 37
```

## 8. Final Tile Result

```text
C00 =
27 17
67 37
```

## 9. What the CUDA Kernel Is Doing

The kernel loop:

```cpp
for (int t = 0; t < numTiles; t++)
```

performs:
1. Load tile of A into shared memory.
2. Load tile of B into shared memory.
3. Compute partial products.
4. Accumulate into `sum`.

Equivalent mathematical form:

```text
C_tile(i,j) += A_tile(i,t) x B_tile(t,j)
```

## 10. Why Tiling Is Faster

Without tiling, each thread repeatedly reads from global memory, which is slow.

With tiling:
- Tiles are loaded once per phase.
- Tiles are stored in shared memory (much faster).
- The same loaded values are reused by many threads.

Example:

```text
Tile size = 16
256 threads reuse the same tile data
```

This dramatically reduces global memory access.

## 11. Key Insight

Instead of computing:

```text
row of A x column of B
```

the tiled algorithm computes:

```text
tile of A x tile of B
```

and accumulates across tile phases.

In a simplified memory-access model, this can reduce effective global reads from roughly:

```text
O(n^3)
```

to approximately:

```text
O(n^3 / TILE_SIZE)
```

which is why tiled matrix multiplication is often much faster than naive GPU implementations.

## 12. Summary

Tiled matrix multiplication works by:
- Dividing matrices into tiles.
- Loading tiles into shared memory.
- Reusing tile data for many multiply-accumulate operations.
- Accumulating partial results across tile phases.

This technique is a foundation of high-performance GPU math libraries such as:
- cuBLAS
- PyTorch
- TensorFlow
