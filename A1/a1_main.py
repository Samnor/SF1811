import numpy as np
from scipy.optimize import linprog


def get_exam_input_april2020():
    new_input = {
        "c":np.array([-2, -1, 0, 0]),
        "A":np.array([
                     [1, 1, 1, 0],
                     [1, -1, 0, 1]
                     ]),
        "b":np.array([3, 1]),
        "solution": True
    }
    return new_input


def get_book_input_page44():
    """
    Example used to show the Simplex Method.
    """
    new_input = {
        "c":np.array([-400, -300, 0, 0]),
        "A":np.array([
                     [1, 1, 1, 0],
                     [2, 1, 0, 1]
                     ]),
        "b":np.array([200, 300]),
        "solution": True
    }
    return new_input


def generate_input(dim_m=2, dim_n=3):
    new_input = {
        "c":np.random.normal(size=dim_n),
        "A":np.random.normal(size=(dim_m, dim_n)),
        "b":np.random.uniform(low=0.01, high=5.0, size=dim_m)
    }
    return new_input


def get_lingprog_answer(form_input):
    return linprog(c=form_input["c"],
                   A_ub=form_input["A"],
                   b_ub=form_input["b"],
                   method="simplex")


def solve_auxilliary_problem_first(*, input_dic):
    print("solve_auxilliary_problem_first")
    def objective_function(*, vector_y):
        return np.sum(vector_y)
    m_dim = input_dic["A"].shape[0]
    n_dim = input_dic["A"].shape[1]
    I_matrix = np.identity(m_dim)
    A_tilde = np.concatenate((input_dic["A"], I_matrix), axis=1)
    print(A_tilde)
    print(A_tilde.shape)

    solution = simplex_solve(matrix_A=A_tilde,
                             vector_b=input_dic["b"],
                             vector_c=input_dic["c"],
                             basic_index_tuple=list(range(m_dim, n_dim)),
                             non_basic_index_tuple=list(range(0, m_dim)))
    print(f"solution {solution}")
    return transform_aux_solution_to_start_solution(aux_solution=solution,
                                                    m_dim=m_dim,
                                                    n_dim=n_dim)


def transform_aux_solution_to_start_solution(*, aux_solution, m_dim, n_dim):
    start_solution = []
    for variable_index, val in enumerate(aux_solution):
        if(val != 0):
            start_solution.append(variable_index)
    assert(len(start_solution) == m_dim)
    return start_solution


def is_vector_semi_positive_definite(test_vector):
    return True if np.sum((test_vector >= 0)) == test_vector.size else False


def is_vector_semi_negative_definite(test_vector):
    return True if np.sum((test_vector <= 0)) == test_vector.size else False


def create_solution_from_b_line(b_line, basic_index_tuple, dim_n):
    sol = np.zeros(dim_n)
    for (b_index, b_val) in zip(basic_index_tuple, b_line.tolist()):
        sol[b_index] = b_val
    return sol


def calc_b_line(*, matrix_A, vector_b, basic_index_tuple):
    print(f"calc_b_line")
    print(f"basic_index_tuple {basic_index_tuple}")
    basic_matrix = matrix_A[:, basic_index_tuple]
    print(f"basic_matrix {basic_matrix}")
    print(f"vector_b {vector_b}")
    print(np.linalg.solve(basic_matrix, vector_b))
    return np.linalg.solve(basic_matrix, vector_b)


def calc_y(*, matrix_A, vector_c, basic_index_tuple):
    basic_matrix = matrix_A[:, basic_index_tuple]
    vector_c_basic = vector_c[basic_index_tuple]
    return np.linalg.solve(basic_matrix.T, vector_c_basic)


def calc_reduced_costs(matrix_A, vector_c, non_basic_index_tuple, vector_y):
    non_basic_matrix = matrix_A[:, non_basic_index_tuple]
    vector_c_non_basic = vector_c[non_basic_index_tuple]
    return vector_c_non_basic - non_basic_matrix.T @ vector_y


def calc_a_line_q(*, q_index, basic_index_tuple, non_basic_index_tuple, matrix_A):
    basic_matrix = matrix_A[:, basic_index_tuple]
    a_v_q = matrix_A[:, non_basic_index_tuple[q_index]]
    return np.linalg.solve(basic_matrix, a_v_q)


def calc_t_max_and_p_index(*, b_line, a_line_q):
    # t_max can actually be skipped
    a_line_q[a_line_q == 0] = 0.000001 # Make certain that these ones are not chosen 
    print(f"b_line {b_line.shape}")
    ratios = [b_line_i/a_line_i_q for (b_line_i, a_line_i_q) in zip(b_line.tolist(), a_line_q.tolist())]
    p_index = np.argmin(ratios)
    return p_index


def simplex_solve(*, matrix_A, vector_b, vector_c, basic_index_tuple,
                  non_basic_index_tuple):
    max_iter = 10000
    dim_n = matrix_A.shape[1]
    iter = 0
    while True:
        print(f"iter {iter}")
        if(iter > max_iter):
            print("simplex_solve passed max_iter, returning False")
            return False
        b_line = calc_b_line(matrix_A=matrix_A,
                             vector_b=vector_b,
                             basic_index_tuple=basic_index_tuple)
        y = calc_y(matrix_A=matrix_A,
                   vector_c=vector_c,
                   basic_index_tuple=basic_index_tuple)
        r_v = calc_reduced_costs(matrix_A=matrix_A,
                                 vector_c=vector_c,
                                 non_basic_index_tuple=non_basic_index_tuple,
                                 vector_y=y)
        if(is_vector_semi_positive_definite(test_vector=r_v)):
            # Optimal solution found
            return create_solution_from_b_line(b_line=b_line,
                                               basic_index_tuple=basic_index_tuple,
                                               dim_n=dim_n)
        q_index = np.argmin(r_v)
        r_v_q = r_v[q_index]
        a_line_q = calc_a_line_q(q_index=q_index,
                                 basic_index_tuple=basic_index_tuple,
                                 non_basic_index_tuple=non_basic_index_tuple,
                                 matrix_A=matrix_A)
        if(is_vector_semi_negative_definite(a_line_q)):
            # No optimal solution for problem
            return False
        #t_max = calc_t_max()
        p_index = calc_t_max_and_p_index(b_line=b_line, a_line_q=a_line_q)
        temp_index = basic_index_tuple[p_index]
        basic_index_tuple[p_index] = non_basic_index_tuple[q_index]
        non_basic_index_tuple[q_index] = temp_index
        iter += 1


def main():
    print("Hello world!")
    new_input = get_exam_input_april2020() #get_book_input_page44()
    print(new_input)
    answer = get_lingprog_answer(new_input)
    print(answer)
    start_solution = solve_auxilliary_problem_first(input_dic=new_input)
    non_basic_index_tuple = list(set(range(0, new_input["A"].shape[1])) - set(start_solution))
    print(f"start_solution {start_solution}")
    final_solution = simplex_solve(matrix_A=new_input["A"],
                                   vector_b=new_input["b"],
                                   vector_c=new_input["c"],
                                   basic_index_tuple=start_solution,
                                   non_basic_index_tuple=non_basic_index_tuple)
    print(f"final_solution {final_solution}")


if __name__ == "__main__":
    main()
