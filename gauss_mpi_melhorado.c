#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

void gaussElimination(double *matrix, double *b, int n, int rank, int size) {
    double *pivot_row = (double *)malloc(n * sizeof(double));

    for (int k = 0; k < n; ++k) {
        // Distribui a linha pivô para todos os processos com MPI_Scatter
        MPI_Scatter(&matrix[k * n], n / size, MPI_DOUBLE, pivot_row, n / size, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        // Reduzir e compartilhar o elemento de b correspondente
        double pivot_b;
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        if (rank == k % size) {
            // Atualiza a linha pivô no processo raiz para manter consistência local
            for (int j = 0; j < n; ++j) {
                matrix[k * n + j] = pivot_row[j];
            }
            b[k] = pivot_b;
        }

        // Eliminação das linhas subsequentes
        for (int i = k + 1; i < n; ++i) {
            if (i % size == rank) {
                double factor = matrix[i * n + k];
                for (int j = k; j < n; ++j) {
                    matrix[i * n + j] -= factor * pivot_row[j];
                }
                b[i] -= factor * pivot_b;
            }
        }
    }

    free(pivot_row);
}

void backSubstitution(double *matrix, double *b, double *x, int n, int rank, int size) {
    for (int i = n - 1; i >= 0; --i) { // Começa da última linha e vai "subindo"
        if (rank == i % size) {
            x[i] = b[i]; // Inicializa x[i] com o valor correspondente em b[i]
            for (int j = i + 1; j < n; ++j) { // Subtrai as contribuições das variáveis já resolvidas
                x[i] -= matrix[i * n + j] * x[j];
            }
            x[i] /= matrix[i * n + i]; // Corrigido: Normalização pelo coeficiente da diagonal principal
        }
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    double tempo_inicial, tempo_final; /* Tempo de execução */

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
            matrix[i * n + j] = (i == j) ? rand() % 100 + 1 : rand() % 100; // Matriz diagonal dominante
        }
    }

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    tempo_inicial = MPI_Wtime();
    gaussElimination(matrix, b, n, rank, size);
    backSubstitution(matrix, b, x, n, rank, size);
    tempo_final = MPI_Wtime();

    if (rank == 0) {
        printf("Foram gastos %.10f segundos\n",tempo_final-tempo_inicial);

        // Exemplo de como consultar soluções
        // for (int i = 0; i < 10 && i < n; ++i) {
        //     printf("x[%d] = %f\n", i, x[i]);
        // }

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
    }

    MPI_Finalize();

    // Liberação de memória
    free(matrix);
    free(b);
    free(x);

    return 0;
}
