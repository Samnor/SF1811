import numpy as np
from scipy.optimize import linprog

def get_book_input_ex5point4():
    new_input = {
        "c":np.array([-3,4,-2,5,0,0]),
        "A":np.array([[1,1,-1,-1,1,0], [1,-1,1,-1,0,1]]),
        "b":np.array([8,4]),
        "solution": False
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

def main():
    print("Hello world!")
    new_input = generate_input()
    print(new_input)
    answer = get_lingprog_answer(new_input)
    print(answer)


if __name__ == "__main__":
    main()