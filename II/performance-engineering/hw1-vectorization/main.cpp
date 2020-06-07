#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <cblas.h>

#include <time.h>

#include <string>
#include <iostream>

#define VEC_LEN 1000000
#define M_LEN 1000
#define RANDOM_SEED 42
#define PACK_SIZE 8

float va[VEC_LEN] __attribute__((aligned(16)));
float vb[VEC_LEN] __attribute__((aligned(16)));
float vc[VEC_LEN] __attribute__((aligned(16)));
float vd[VEC_LEN] __attribute__((aligned(16)));

float vec_res[2 * PACK_SIZE] __attribute__((aligned(16)));

double mA[M_LEN][M_LEN];
double mB[M_LEN][M_LEN];
double mB_T[M_LEN][M_LEN];
double mC[M_LEN][M_LEN];

double m_vec_res[VEC_LEN * PACK_SIZE] __attribute__((aligned(16)));

std::string hay;
std::string needle;

const char *c_hay;
const char *c_needle;

size_t n;
size_t k;

/*
** Generates random float number from 0 to 1
*/
float rand_float() {
    return (float)rand() / (float)(RAND_MAX);
}

int nothing_vec(float va[VEC_LEN], float vb[VEC_LEN], float vc[VEC_LEN]) {
    return 0;
}

/*
** Run a function N times and calculate overall time
*/
void calc_function_time(void (*f)(), const char *f_name, const size_t num) {
    double  time_spent;
    clock_t begin;
    clock_t end;

    begin = clock();
    for (size_t i = 0; i < num; ++i) {
        f();
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%s, %zu iterations, time: %f\n", f_name, num, time_spent);
}

/*
** Initialize vector with random float numbers within range from 0. to 1000.
*/
void initialize_vectors() {
    for (size_t i = 0; i < VEC_LEN; ++i) {
        va[i] = rand_float() * (float)10.;
        vb[i] = rand_float() * (float)10.;
        vc[i] = rand_float() * (float)10.;
        vd[i] = rand_float() * (float)10.;
    }
}


/*
** Multiply 2 vectors (basic version)
*/
void vector_mult_basic() {
    float res = 0;

    for (int i = 0; i < VEC_LEN; ++i) {
        res += va[i] * vb[i] + vc[i] * vd[i];
    }
}

/*
** Multiply 2 vectors using intrinsics
*/
void vector_mult_vectorized() {
    __m256 rA, rB, rC, rD, rRab, rRcd;
    float   res = 0;

    rRab = _mm256_setzero_ps();
    rRcd = _mm256_setzero_ps();

    for (size_t i = 0; i < VEC_LEN; i += PACK_SIZE) {
        rA = _mm256_loadu_ps(&va[i]);
        rB = _mm256_loadu_ps(&vb[i]);
        rC = _mm256_loadu_ps(&vc[i]);
        rD = _mm256_loadu_ps(&vd[i]);

        rRab = _mm256_fmadd_ps(rA, rB, rRab);
        rRcd = _mm256_fmadd_ps(rC, rD, rRcd);
    }
    _mm256_storeu_ps(&vec_res[0], rRab);
    _mm256_storeu_ps(&vec_res[PACK_SIZE], rRcd);

    for (size_t i = 0; i < 2 * PACK_SIZE; ++i) {
        res += vec_res[i];
    }
    nothing_vec(va, vb, vc);
}

/*
** Initialize matrices with random float numbers within range from 0. to 1000.
*/
void initialize_matrices() {
    for (size_t i = 0; i < M_LEN; ++i) {
        for (size_t j = 0; j < M_LEN; ++j) {
            mA[i][j] = rand_float() * (float)10.;
            mB[i][j] = rand_float() * (float)10.;
            mC[i][j] = 0;
        }
    }
}

/*
** Transpose function so we can benefit from row cache
*/
void transpose_matrix() {
    for (size_t i = 0; i < M_LEN; ++i) {
        for (size_t j = 0; j < M_LEN; ++j) {
            mB_T[i][j] = mB[j][i];
        }
    }
}

/*
** Basic implementation of matrix multiplication
*/
void multiply_matrices_basic() {
    for (size_t i = 0; i < M_LEN; ++i) {
        for (size_t j = 0; j < M_LEN; ++j) {
            mC[i][j] = 0;
            for (size_t k = 0; k < M_LEN; ++k) {
                mC[i][j] += mA[i][k] * mB[k][j];
            }
        }
    }
}

/*
** Generate random string
*/
std::string random_string(size_t len) {
    auto randchar = []() -> char
    {
        const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };
    std::string str(len, 0);
    std::generate_n(str.begin(), len, randchar);

    return str;
}

/*
** Generate random substring
*/
std::string random_substring(std::string str, size_t len) {
    const size_t random_index = rand() % (str.length() - len - 2);
    std::string substring = str.substr(random_index, random_index + len);

    return substring;
}

/*
** Initialize strings
*/
void initialize_strings(size_t str_len, size_t substr_len) {
    hay = random_string(str_len);
    needle = random_substring(hay, substr_len);

    c_hay = hay.c_str();
    c_needle = needle.c_str();

    n = hay.length();
    k = needle.length();

}
double multiply_vectors_from_matrices(int i, int j) {
    __m256 rA, rB, rR;
    double   res = 0;

    rR = _mm256_setzero_ps();

    for (size_t k = 0; k < M_LEN; k += PACK_SIZE) {
        rA = _mm256_loadu_pd(&mA[i][k]);
        rB = _mm256_loadu_pd(&mB_T[j][k]);

        rR = _mm256_fmadd_ps(rA, rB, rR);
    }
    _mm256_storeu_pd(&m_vec_res[0], rR);

    for (size_t k = 0; k < PACK_SIZE; ++k) {
        res += m_vec_res[k];
    }

    return res;
}

/*
** Basic implementation of matrix multiplication
*/
void multiply_matrices_vectorized() {
    transpose_matrix();

    for (size_t i = 0; i < M_LEN; ++i) {
        for (size_t j = 0; j < M_LEN; ++j) {
            mC[i][j] = multiply_vectors_from_matrices(i, j);
        }
    }
}

/*
** Multiply matricues with OpenBlast
*/
void openblas_matrix_mult() {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            (int) M_LEN, (int) M_LEN, (int) M_LEN, 1,
            &mA[0][0], (int) M_LEN, &mB[0][0], (int) M_LEN, (int) M_LEN, &mC[0][0], (int) M_LEN);
}

/*
** Basic substring search
*/
void basic_substr() {
    std::size_t pos = hay.find(needle);
}

/*
** Vectorized substring search
** http://0x80.pl/articles/simd-strfind.html
*/
void vectorized_substr() {
    const __m256i first = _mm256_set1_epi8(c_needle[0]);
    const __m256i last  = _mm256_set1_epi8(c_needle[k - 1]);

    for (size_t i = 0; i < n; i += 32) {

        const __m256i block_first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_hay + i));
        const __m256i block_last  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_hay + i + k - 1));

        const __m256i eq_first = _mm256_cmpeq_epi8(first, block_first);
        const __m256i eq_last  = _mm256_cmpeq_epi8(last, block_last);

        uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(eq_first, eq_last));

        while (mask != 0) {
            const auto bitpos = __builtin_ctzl(mask);

            if (memcmp(c_hay + i + bitpos + 1, c_needle + 1, k - 2) == 0) {
                return;
            }

            mask &= mask - 1;
        }
    }
}

int main() {
    /*
    ** Task 1
    ** A*B + C*D, де A,B,C,D - вектори однакової довжини (масиви)
    */
    printf("Task 1: Vector multiplication\n");
    srand((unsigned)RANDOM_SEED);

    initialize_vectors();

    calc_function_time(&vector_mult_basic, "vector_mult_basic", 50);
    calc_function_time(&vector_mult_vectorized, "vector_mult_vectorized", 50);

    /*
    ** Task 2
    ** Множення матриць. Для спрощення обрати квадратні матриці, яки можуть бути розміщені в пам'яті
    */
    printf("\nTask 2: Matrix multiplication\n");
    initialize_matrices();

    calc_function_time(&multiply_matrices_basic, "multiply_matrices_basic", 50);
    calc_function_time(&multiply_matrices_vectorized, "multiply_matrices_vectorized", 50);
    calc_function_time(&openblas_matrix_mult, "openblas_matrix_mult", 50);

    /*
    ** Task 3
    ** Пошук підрядка в рядку. Знайти перше співпадіння
    */
    printf("\nTask 3: Substring search\n");
    initialize_strings(150000000, 30);

    calc_function_time(&basic_substr, "basic_substr", 10);
    calc_function_time(&vectorized_substr, "vectorized_substr", 10);
}
