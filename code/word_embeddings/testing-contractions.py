from pprint import pprint

import contractions


def run_test():
    test_strs = [
        "İlxıçılar (also, Ilxıçılar and Ilkhychylar) is a village in the Tartar Rayon of Azerbaijan.",
        'Isa Yunis oglu Qambar (Azerbaijani: "İsa Yunis oğlu Qəmbər / Иса Jунис оғлу Гәмбәр"), also known as Isa Gambar or Isa Qambar (born February 24, 1957), is a prominent Azerbaijani politician and leader of the Equality Party "(Müsavat)", the largest opposition block in Azerbaijan.',
        "Qarabağlılar is a village in the municipality of İsakənd in the Tovuz Rayon of Azerbaijan.",
        "Məşədismayıllı (also, Məşədi İsmayıllı, Meshadi-Ismaili, Meshadiismailly, and Meshadi-Ismayli) is a village in the Zangilan Rayon of Azerbaijan.",
        "İlxıçı (also, Ilxıçı, Gasan Efendi, Həsən Əfəndi, Il’khychy-Gasanefendi, Il’kichi-Kazeya, and Ilkhychy) is a village and municipality in the Khachmaz Rayon of Azerbaijan.",
    ]
    all_unique_chars = set()
    for test_str in test_strs:
        for c in test_str.replace(" ", ""):
            all_unique_chars.add(c)
    pprint(sorted(all_unique_chars))


def run_test_2():
    print(contractions.fix("İ feb."))
    print(
        contractions.fix(
            "Qarabağlılar is a village in the municipality of İsakənd in the Tovuz Rayon of Azerbaijan."
        )
    )


if __name__ == "__main__":
    run_test()
    run_test_2()
