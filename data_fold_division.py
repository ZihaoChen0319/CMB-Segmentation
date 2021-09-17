import json


if __name__ == '__main__':
    division = {"0": {"train": ["sub-101", "sub-102", "sub-103", "sub-104", "sub-106", "sub-107", "sub-108", "sub-109",
                                "sub-110", "sub-111", "sub-201", "sub-202", "sub-203", "sub-204", "sub-205", "sub-206",
                                "sub-207", "sub-208", "sub-209", "sub-211", "sub-213", "sub-214", "sub-215", "sub-217",
                                "sub-219", "sub-221", "sub-222", "sub-223", "sub-224", "sub-225", "sub-226", "sub-227",
                                "sub-228", "sub-229", "sub-231", "sub-232", "sub-233", "sub-234", "sub-301", "sub-302",
                                "sub-303", "sub-306", "sub-308", "sub-309", "sub-310", "sub-311", "sub-314", "sub-315",
                                "sub-316", "sub-317", "sub-319", "sub-322", "sub-323", "sub-324", "sub-325", "sub-326",
                                "sub-327"],
                      "val": ["sub-105", "sub-210", "sub-212", "sub-216", "sub-218", "sub-220", "sub-230", "sub-304",
                              "sub-305", "sub-307", "sub-312", "sub-313", "sub-318", "sub-320", "sub-321"]},

                "1": {"train": ["sub-101", "sub-102", "sub-103", "sub-104", "sub-105", "sub-106", "sub-108", "sub-110",
                                "sub-111", "sub-201", "sub-203", "sub-204", "sub-205", "sub-206", "sub-207", "sub-208",
                                "sub-210", "sub-211", "sub-212", "sub-213", "sub-216", "sub-217", "sub-218", "sub-219",
                                "sub-220", "sub-221", "sub-223", "sub-224", "sub-226", "sub-227", "sub-228", "sub-229",
                                "sub-230", "sub-231", "sub-233", "sub-302", "sub-304", "sub-305", "sub-306", "sub-307",
                                "sub-308", "sub-310", "sub-311", "sub-312", "sub-313", "sub-315", "sub-316", "sub-317",
                                "sub-318", "sub-319", "sub-320", "sub-321", "sub-322", "sub-324", "sub-325", "sub-326",
                                "sub-327"],
                      "val": ["sub-107", "sub-109", "sub-202", "sub-209", "sub-214", "sub-215", "sub-222", "sub-225",
                              "sub-232", "sub-234", "sub-301", "sub-303", "sub-309", "sub-314", "sub-323"]},

                "2": {"train": ["sub-102", "sub-105", "sub-107", "sub-108", "sub-109", "sub-110", "sub-111", "sub-201",
                                "sub-202", "sub-203", "sub-204", "sub-206", "sub-207", "sub-208", "sub-209", "sub-210",
                                "sub-212", "sub-213", "sub-214", "sub-215", "sub-216", "sub-217", "sub-218", "sub-219",
                                "sub-220", "sub-221", "sub-222", "sub-224", "sub-225", "sub-226", "sub-228", "sub-229",
                                "sub-230", "sub-231", "sub-232", "sub-233", "sub-234", "sub-301", "sub-303", "sub-304",
                                "sub-305", "sub-306", "sub-307", "sub-309", "sub-310", "sub-311", "sub-312", "sub-313",
                                "sub-314", "sub-315", "sub-318", "sub-320", "sub-321", "sub-322", "sub-323", "sub-324",
                                "sub-326", "sub-327"],
                      "val": ["sub-101", "sub-103", "sub-104", "sub-106", "sub-205", "sub-211", "sub-223", "sub-227",
                              "sub-302", "sub-308", "sub-316", "sub-317", "sub-319", "sub-325"]},

                "3": {"train": ["sub-101", "sub-102", "sub-103", "sub-104", "sub-105", "sub-106", "sub-107", "sub-108",
                                "sub-109", "sub-202", "sub-204", "sub-205", "sub-206", "sub-209", "sub-210", "sub-211",
                                "sub-212", "sub-214", "sub-215", "sub-216", "sub-217", "sub-218", "sub-220", "sub-221",
                                "sub-222", "sub-223", "sub-224", "sub-225", "sub-226", "sub-227", "sub-230", "sub-231",
                                "sub-232", "sub-234", "sub-301", "sub-302", "sub-303", "sub-304", "sub-305", "sub-307",
                                "sub-308", "sub-309", "sub-310", "sub-311", "sub-312", "sub-313", "sub-314", "sub-316",
                                "sub-317", "sub-318", "sub-319", "sub-320", "sub-321", "sub-323", "sub-324", "sub-325",
                                "sub-326", "sub-327"],
                      "val": ["sub-110", "sub-111", "sub-201", "sub-203", "sub-207", "sub-208", "sub-213", "sub-219",
                              "sub-228", "sub-229", "sub-233", "sub-306", "sub-315", "sub-322"]},  # "sub-110" outlier

                "4": {"train": ["sub-101", "sub-103", "sub-104", "sub-105", "sub-106", "sub-107", "sub-109", "sub-110",
                                "sub-111", "sub-201", "sub-202", "sub-203", "sub-205", "sub-207", "sub-208", "sub-209",
                                "sub-210", "sub-211", "sub-212", "sub-213", "sub-214", "sub-215", "sub-216", "sub-218",
                                "sub-219", "sub-220", "sub-222", "sub-223", "sub-225", "sub-227", "sub-228", "sub-229",
                                "sub-230", "sub-232", "sub-233", "sub-234", "sub-301", "sub-302", "sub-303", "sub-304",
                                "sub-305", "sub-306", "sub-307", "sub-308", "sub-309", "sub-312", "sub-313", "sub-314",
                                "sub-315", "sub-316", "sub-317", "sub-318", "sub-319", "sub-320", "sub-321", "sub-322",
                                "sub-323", "sub-325"],
                      "val": ["sub-102", "sub-108", "sub-204", "sub-206", "sub-217", "sub-221", "sub-224", "sub-226",
                              "sub-231", "sub-310", "sub-311", "sub-324", "sub-326", "sub-327"]},

                "overall": ["sub-101", "sub-102", "sub-103", "sub-104", "sub-105", "sub-106", "sub-107", "sub-108",
                            "sub-109", "sub-110", "sub-111", "sub-201", "sub-202", "sub-203", "sub-204", "sub-205",
                            "sub-206", "sub-207", "sub-208", "sub-209", "sub-210", "sub-211", "sub-212", "sub-213",
                            "sub-214", "sub-215", "sub-216", "sub-217", "sub-218", "sub-219", "sub-220", "sub-221",
                            "sub-222", "sub-223", "sub-224", "sub-225", "sub-226", "sub-227", "sub-228", "sub-229",
                            "sub-230", "sub-231", "sub-232", "sub-233", "sub-234", "sub-301", "sub-302", "sub-303",
                            "sub-304", "sub-305", "sub-306", "sub-307", "sub-308", "sub-309", "sub-310", "sub-311",
                            "sub-312", "sub-313", "sub-314", "sub-315", "sub-316", "sub-317", "sub-318", "sub-319",
                            "sub-320", "sub-321", "sub-322", "sub-323", "sub-324", "sub-325", "sub-326", "sub-327"]}

    with open('./dataset/fold_division.json', mode='w') as f:
        json.dump(division, f)


