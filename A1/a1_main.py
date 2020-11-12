import numpy as np
from scipy.optimize import linprog


def get_exam_input_april2020():
    # Canonical form
    new_input = {
        "c":np.array([-2, -1]),
        "A":np.array([
                     [1, 1],
                     [1, -1]
                     ]),
        "b":np.array([3, 1]),
        "solution": True
    }
    return new_input


def get_exam_input_jan2020():
    # Canonical form
    new_input = {
        "c":np.array([-10, -6, 8]),
        "A":np.array([
                     [5, -2, 6],
                     [10, 4, -6]
                     ]),
        "b":np.array([20, 30]),
        "solution": True
    }
    return new_input


def get_book_input_page44():
    """
    Example used to show the Simplex Method.
    Canonical form

    """
    new_input = {
        "c":np.array([-400, -300]),
        "A":np.array([
                     [1, 1],
                     [2, 1]
                     ]),
        "b":np.array([200, 300]),
        "solution": True
    }
    return new_input


def get_book_input_ex54():
    # Canonical form
    new_input = {
        "c":np.array([-3, 4, -2, 5]),
        "A":np.array([
                     [1, 1, -1, -1],
                     [1, -1, 1, -1]
                     ]),
        "b":np.array([8, 4]),
        "solution": True
    }
    return new_input


def get_book_input_ex54_alternate():
    # Canonical form
    new_input = {
        "c":np.array([-3, 4, -2, 2]),
        "A":np.array([
                     [1, 1, -1, -1],
                     [1, -1, 1, -1]
                     ]),
        "b":np.array([8, 4]),
        "solution": False
    }
    return new_input


def get_linprog_answer(form_input):
    return linprog(c=form_input["c"],
                   A_eq=form_input["A"],
                   b_eq=form_input["b"],
                   method="simplex")


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
    basic_matrix = matrix_A[:, basic_index_tuple]
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
    ratios = []
    LARGE_VALUE = 999999
    for (b_line_i, a_line_i_q) in zip(b_line.tolist(), a_line_q.tolist()):
        new_ratio = b_line_i/a_line_i_q
        if(a_line_i_q > 0 and new_ratio > 0):
            ratios.append(new_ratio)
        else:
            ratios.append(LARGE_VALUE)
    p_index = np.argmin(ratios)
    return p_index


def simplex_solve(*, matrix_A, vector_b, vector_c, basic_index_tuple,
                  non_basic_index_tuple):
    max_iter = 100000
    dim_n = matrix_A.shape[1]
    iter = 0
    # This loop keeps the iterations of the simplex method going until we have found a solution
    # We use a max_iter variable to avoid infinite loops during testing
    while True:
        # Run simplex iteration {iter}
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
        p_index = calc_t_max_and_p_index(b_line=b_line, a_line_q=a_line_q)
        temp_index = basic_index_tuple[p_index]
        print(f"Replacing {temp_index} with {non_basic_index_tuple[q_index]} in basic_index_tuple")
        basic_index_tuple[p_index] = non_basic_index_tuple[q_index]
        non_basic_index_tuple[q_index] = temp_index
        iter += 1


def add_slack_variables(new_input):
    m_dim = new_input["A"].shape[0]
    n_dim = new_input["A"].shape[1]
    I_matrix = np.identity(m_dim)
    A_tilde = np.concatenate((new_input["A"], I_matrix), axis=1)
    c_tilde = np.concatenate((new_input["c"],np.zeros(m_dim)))
    new_input["A"] = A_tilde
    new_input["c"] = c_tilde
    return new_input


def test_simplex_implementation():
    # Below is a list of functions to retrieve simplex problems from the course
    input_list = [get_exam_input_jan2020,
                  get_exam_input_april2020,
                  get_book_input_page44,
                  get_book_input_ex54,
                  get_book_input_ex54_alternate]
    # This loops tests our simplex implementation on every simplex example above
    for input_source in input_list:
        new_input = input_source()
        new_input = add_slack_variables(new_input) # Transform canonical form to Standard form with slack variables
        answer = get_linprog_answer(new_input) # This is our benchmark.
        print(f"answer {answer}")
        m_dim = new_input["A"].shape[0]
        n_dim = new_input["A"].shape[1]
        col_indices = list(range(0, n_dim))
        slack_basic_index_tuple = col_indices[-1*m_dim:]
        non_basic_index_tuple = col_indices[0:-1*m_dim]
        final_solution = simplex_solve(matrix_A=new_input["A"],
                                    vector_b=new_input["b"],
                                    vector_c=new_input["c"],
                                    basic_index_tuple=slack_basic_index_tuple,
                                    non_basic_index_tuple=non_basic_index_tuple)
        if(final_solution is False):
            print("No solution found")
        else:
            print(f"final_solution {final_solution}")


def main():
    test_simplex_implementation()


if __name__ == "__main__":
    main()
