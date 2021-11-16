import os
import json


def main():
    """
    Group   # src # tar # Formula
    ------------------------------
    CA      1       68      RCOO
    Ester   1       7       RCOOR'
    Ketone  1       15      ROR'
    Phenyl  22      36      Aromatic Rings
    Tbutyl  1       10      C4
    dsAmide 4       18      RONR'R"
    msAmide 2       32      RONR'
    nsAmide 4       32      RON
    """
    
    with open('isostere_transformations_new.json', 'r') as fp:
        rules_json = json.load(fp)
    
    types = {
        'Add_CH*', 'Add_CH*CH*', 'Add_CH*CH*CH*',
        'CA', 'CAI',
        'Drop_CH*', 'Drop_CH*CH*', 'Drop_CH*CH*CH*',
        'EI', 'Ester',
        'KI', 'Ketone',
        'P', 'PI', 'Phenyl', 'TBI',
        'Tbutyl', 'dsAI', 'dsAmide', 'msAI', 'msAmide', 'nsAI', 'nsAmide'
    }
    tags = {
        'Acid Isostere',
        'Add C',
        'Amide Isostere',
        'Disubstituted Amide Isostere',
        'Drop C',
        'Ester Isostere',
        'Isostere',
        'Ketone Isostere',
        'Monosubstituted Amide Isostere',
        'Phenyl Isostere',
        'T-butyl Isostere',
        'Unsubstituted Amide Isostere'
    }

    rules = {k: [] for k in types}
    


if __name__ == '__main__':
    main()
