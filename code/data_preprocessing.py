import pandas as pd
from pyswip import Prolog



def test_prolog(prolog_path):
    query_str1 = "country(ph, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)"
    query_str2 = "country(ph, Results)"
    query_str3 = "person( _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ )"


    prolog = Prolog()
    prolog.consult(prolog_path)
    solutions1 = prolog.query(query_str1)
    solutions2 = list(prolog.query(query_str2))[0]["Results"]
    solutions3 = prolog.query(query_str3)
    
    print("Solutions2 num results: ", len(solutions2))

    i = 0    
    for sol in solutions1:
        i += 1
    
    print("Solutions1 num results: ", i)

    i = 0    
    for sol in solutions3:
        i += 1
    
    print("Solutions3 num results: ", i)

    

def read_csv(path, col_to_drop = []):
    data_file = pd.read_csv(path, delimiter="\t")

    for col in col_to_drop:
        try:
            data_file.drop(col, axis = 1, inplace=True)
            print(f"[+] Column {col} removed")
        except Exception as e:
            print(f"[-] {e}")
            continue

    return data_file


def shrink_data(data_file):
    data_copy = data_file.copy(deep = True)
    question_letters = [chr(i) for i in range(ord('A'), ord('P') + 1)]
    question_nums = {"A":10, "B":13, "C":10, "D":10, "E":10, "F":10,
                     "G":10, "H":10, "I":10, "J":10, "K":10, "L":10, 
                     "M":10, "N":10, "O":10, "P":10 }
    question_partitions = []
    for letter in question_letters:
        partition = [f"{letter}{num}" for num in range(1, question_nums[letter]+1)]
        question_partitions.append(partition)

    for i in range(len(question_letters)):
        norm_factor = 5*question_nums[question_letters[i]]
        data_copy[question_letters[i]] = round(data_copy[question_partitions[i]].sum(axis = 1)/norm_factor, 2)
        data_copy.drop(question_partitions[i], axis = 1, inplace = True)

    return data_copy


def rows_to_facts(data_file):    
    cols = data_file.columns.tolist()
    cols_upper = [col.upper() for col in cols]

    facts = []
    fact_head = "person( "
    fact_tail = " )."


    for index, row in data_file.iterrows():
        fact = fact_head
        for i in range(len(cols)):
            if i == len(cols)-1:
                fact += f"{row[cols[i]]}".lower()
            else:
                fact += f"{row[cols[i]]}, ".lower()
        fact += fact_tail
        facts.append(fact)
    
    return facts


def write_prolog(lines, prolog_path):
    try:
        with open(prolog_path, mode="w") as pf:
            for line in lines:
                pf.write(line + "\n")
    except Exception as e:
        print(f"{e}")



def main():
    rules = ["country(COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) :- person( _, _, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), COUNTRY = COUNTRY1.",
            "country(COUNTRY, Results) :- findall(person( GENDER1, ACCURACY1, COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), person( GENDER1, ACCURACY1, COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), Results).",
            "gender(GENDER, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) :- person( GENDER1, _, _, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), GENDER1 = GENDER.",
            "gender(GENDER, Results) :- findall(person( GENDER, ACCURACY1, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), person( GENDER, ACCURACY1, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), Results)."
    ]
    data_path = "../data/16PF/data.csv"
    prolog_path = "data.pl"
    small_prolog_path = "small_data.pl"
    to_drop = ["source", "elapsed", "age"]

    print("[*] Reading csv file...")

    df = read_csv(data_path, to_drop)

    print("[+] csv file read correctly")
    
    print("[*] Shrinking data...")

    df = shrink_data(df)

    print("[*] Converting data in prolog clauses...")

    facts = rows_to_facts(df)
    program = rules + facts

    print("[*] Writing program to prolog file...")

    write_prolog(program, small_prolog_path)

    print("[+] Program file written correctly")

    print("[*] Testing prolog query...")

    test_prolog(small_prolog_path)
    
    


if __name__ == "__main__":
    main()