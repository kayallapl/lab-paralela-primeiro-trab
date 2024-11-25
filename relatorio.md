O chatGPT forneceu o seguinte código:

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void gaussElimination(double *matrix, double *b, int n, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        if (rank == k % size) {
            // Normalizar linha pivô
            double pivot = matrix[k * n + k];
            for (int j = k; j < n; ++j) {
                matrix[k * n + j] /= pivot;
            }
            b[k] /= pivot;
        }

        // Broadcast da linha pivô
        MPI_Bcast(&matrix[k * n], n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        // Eliminação das linhas subsequentes
        for (int i = k + 1; i < n; ++i) {
            if (i % size == rank) {
                double factor = matrix[i * n + k];
                for (int j = k; j < n; ++j) {
                    matrix[i * n + j] -= factor * matrix[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }
    }
}

void backSubstitution(double *matrix, double *b, double *x, int n, int rank, int size) {
    for (int i = n - 1; i >= 0; --i) {
        if (rank == i % size) {
            x[i] = b[i];
            for (int j = i + 1; j < n; ++j) {
                x[i] -= matrix[i * n + j] * x[j];
            }
        }
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    int n = 3; // Dimensão do sistema (modifique conforme necessário)
    double matrix[] = {
        2.0, 1.0, -1.0,
        -3.0, -1.0, 2.0,
        -2.0, 1.0, 2.0
    };
    double b[] = {8.0, -11.0, -3.0};
    double x[3];

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gaussElimination(matrix, b, n, rank, size);
    backSubstitution(matrix, b, x, n, rank, size);

    if (rank == 0) {
        printf("Solução do sistema:\n");
        for (int i = 0; i < n; ++i) {
            printf("x[%d] = %f\n", i, x[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
```

A primeira coisa que fizemos foi adicionar a função MPI_Wtime() para marcar o tempo de execução.

```c
tempo_inicial = MPI_Wtime();
gaussElimination(matrix, b, n, rank, size);
backSubstitution(matrix, b, x, n, rank, size);
tempo_final = MPI_Wtime();
```

Os tempos iniciais foram muito pequenos, porém os tempos para mais processos estava dando maior. Isso se dá pelo overhead de comunicação, a própria divisão de processos acabou usando mais tempo:

```shell
$ mpirun -np 2 ./gauss_mpi_melhorado
Solução do sistema:
x[0] = 2.000000
x[1] = 3.000000
x[2] = -1.000000
Foram gastos 0.0000430000 segundos

$ mpirun -np 4 ./gauss_mpi_melhorado
Solução do sistema:
x[0] = 2.000000
x[1] = 3.000000
x[2] = -1.000000
Foram gastos 0.0000680000 segundos

$ mpirun -np 8 ./gauss_mpi_melhorado
Solução do sistema:
x[0] = 2.000000
x[1] = 3.000000
x[2] = -1.000000
Foram gastos 0.0001970000 segundos
```

Então, ao invés de criar uma pequena matriz estática, criamos dinamicamente uma matriz de 2000 valores (para os parâmetros do computador em que foi rodado, é um valor médio). Para isso, usamos malloc para armazenar essa matriz.

```c
    int n = 2000; // Tamanho do sistema
    double *matrix, *b, *x;

    // Alocação dinâmica dos endereços para a matriz
    matrix = (double *)malloc(n * n * sizeof(double));
    b = (double *)malloc(n * sizeof(double));
    x = (double *)malloc(n * sizeof(double));

    if (!matrix || !b || !x) {
        printf("Erro na alocação de memória\n");
        exit(1);
    }
    
    // Preenche matriz e vetor com valores aleatórios
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        b[i] = rand() % 100; // Valores no vetor b
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = (i == j) ? rand() % 100 + 1 : rand() % 100;
            // Matriz diagonal dominante
        }
    }
```

Para conferir se o resultado encontrado para a matriz x está correto, comparamos ax (matriz inicial * resultado encontrado) com o b aleatório inicial:

```c
// Verificando a precisão da solução
double error = 0.0;
for (int i = 0; i < n; ++i) {
    double ax = 0.0;
    for (int j = 0; j < n; ++j) {
        ax += matrix[i * n + j] * x[j]; // calcula os resultados para b com x encontrado
    }
    error += fabs(ax - b[i]); // compara o valor com o b aleatório inicial
}
printf("Erro total: %e\n", error);
```

Obtivemos os seguintes resultados:

```shell
$ mpirun -np 2 ./gauss_mpi_melhorado                
Foram gastos 2.0763030000 segundos
Erro total: 2.814443e-10

$ mpirun -np 4 ./gauss_mpi_melhorado                
Foram gastos 1.1009420000 segundos
Erro total: 1.766998e-10

$ mpirun -np 8 ./gauss_mpi_melhorado                
Foram gastos 2.2496170000 segundos
Erro total: 6.402490e-10
```

Notamos que a função `backSubstitution` tinha um erro: no final, o valor de x[i] não estava sendo normalizado pelo coeficiente da diagonal principal, o que pode levar a resultados incorretos. Adicionamos então a linha `x[i] /= matrix[i * n + i];` para consertar. Isso diminuiu um pouco o erro:

```shell
$ mpirun -np 2 ./gauss_mpi_melhorado                
Foram gastos 2.0777840000 segundos
Erro total: 1.280451e-10

$ mpirun -np 4 ./gauss_mpi_melhorado
Foram gastos 1.1075860000 segundos
Erro total: 1.769355e-10

$ mpirun -np 8 ./gauss_mpi_melhorado
Foram gastos 2.1546820000 segundos
Erro total: 3.593319e-10
```

Depois disso, tentamos rodar para um valor um pouco mais extremo, uma matriz de n = 5.000. O tempo aumentou consideravelmente.

Como o computador era quadcore, tomamos a liberdade de ao invés de usar 2, 4 e 8, usar 1, 2 e 4, pois nos entregava resultados mais satisfatórios em relação ao tempo.

##### Resultado para n = 100:
```shell
$ mpirun -np 1 ./gauss_mpi_melhorado
Foram gastos 0.0012420000 segundos
Erro total: 3.888513e-13

$ mpirun -np 2 ./gauss_mpi_melhorado
Foram gastos 0.0017670000 segundos
Erro total: 1.271587e-13

$ mpirun -np 4 ./gauss_mpi_melhorado
Foram gastos 0.0017940000 segundos
Erro total: 5.108414e-13
```

##### Resultado para n = 2000:
```shell
$ mpirun -np 1 ./gauss_mpi_melhorado                
Foram gastos 3.9990370000 segundos
Erro total: 2.640805e-10

$ mpirun -np 2 ./gauss_mpi_melhorado                
Foram gastos 2.0781060000 segundos
Erro total: 2.371149e-10

$ mpirun -np 4 ./gauss_mpi_melhorado                
Foram gastos 1.1092260000 segundos
Erro total: 9.125933e-11
```

##### Resultado para n = 5000:
```shell
$ mpirun -np 1 ./gauss_mpi_melhorado
Foram gastos 62.5402540000 segundos
Erro total: 6.083387e-10

$ mpirun -np 2 ./gauss_mpi_melhorado
Foram gastos 32.6793270000 segundos
Erro total: 4.035896e-09

$ mpirun -np 4 ./gauss_mpi_melhorado
Foram gastos 17.2685780000 segundos
Erro total: 3.014510e-09
```

