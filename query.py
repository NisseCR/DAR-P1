import re


def main():
    dict = parse()


def parse():
    prompt = input()
    terminatedRE = re.compile(r"[^;]+")
    terminated = terminatedRE.match(prompt).group()
    print(terminated)
    predicatesRE = re.compile(r"[^,]+")
    predicates = predicatesRE.findall(terminated)
    print(predicates)
    dict = {}
    for predicate in predicates:
        attribute, value = predicate.split(" = ")
        attribute = attribute.strip()
        value = eval(value.strip())
        dict[attribute] = value
    if 'k' not in dict:
        dict['k'] = 10
    return dict


if __name__== '__main__':
    main()
