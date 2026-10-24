import unittest

import joblib
import pandas as pd

from src.pipeline import (DATA_PATH, FEATURES, MODEL_PATH, TARGET,
                         audit_dataset, load_and_validate, parse_courses,
                         profile_to_frame, recommend, split_data)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python main.py --step train before this suite.")
        cls.bundle = joblib.load(MODEL_PATH)
        cls.profile = {"Current_Course": "Math101", "Learning_Level": "Beginner",
                       "Learning_Preference": "Text", "Time_Spent_Hours": 20,
                       "Performance_Score": 70, "Completed_Courses": []}

    def test_real_dataset_and_target(self):
        df, audit = audit_dataset()
        self.assertEqual(DATA_PATH.name, audit["file_name"])
        self.assertTrue(audit["target_exists"])
        self.assertEqual(len(df), 1000)
        self.assertEqual(df[TARGET].nunique(), 12)
        self.assertEqual(audit["duplicate_rows"], 0)

    def test_source_schema_and_identifier_exclusion(self):
        df = load_and_validate()
        self.assertTrue(set(FEATURES + [TARGET]).issubset(df.columns))
        self.assertNotIn("Learner_ID", self.bundle["features"])
        self.assertNotIn(TARGET, self.bundle["features"])

    def test_course_history_parse_but_exclude_from_model_features(self):
        self.assertEqual(parse_courses(" Bio102, Math102, Bio102 "), ["Bio102", "Math102"])
        self.assertNotIn("Completed_Courses", self.bundle["features"])
        self.assertIn("Completed_Courses", load_and_validate().columns)

    def test_known_target_history_overlap_is_reported(self):
        _, audit = audit_dataset()
        self.assertEqual(audit["leakage_checks"]["target_in_completed_course_history_rows"], 217)
        self.assertEqual(audit["leakage_checks"]["target_matches_current_course_rows"], 83)

    def test_profile_schema(self):
        row = profile_to_frame(self.profile, self.bundle["features"])
        self.assertEqual(row.columns.tolist(), FEATURES)
        self.assertNotIn("Completed_Courses", row.columns)

    def test_split_is_stratified_and_disjoint(self):
        df = load_and_validate()
        train, test = split_data(df, save=False)
        self.assertEqual(len(train), 800)
        self.assertEqual(len(test), 200)
        self.assertTrue(set(train.Learner_ID).isdisjoint(set(test.Learner_ID)))
        self.assertTrue(set(train[TARGET].unique()) == set(test[TARGET].unique()))

    def test_recommendations_exclude_current_and_completed(self):
        profile = dict(self.profile, Completed_Courses=["Bio102"])
        got = recommend(profile, 5, self.bundle)
        self.assertLessEqual(len(got), 5)
        self.assertTrue(got["Ranking score"].between(0, 1).all())
        self.assertNotIn("Math101", got["Recommended course"].tolist())
        self.assertNotIn("Bio102", got["Recommended course"].tolist())
        self.assertEqual(got["Rank"].tolist(), list(range(1, len(got) + 1)))

    def test_invalid_k_and_no_eligible_courses(self):
        with self.assertRaises(ValueError):
            recommend(self.profile, 6, self.bundle)
        labels = self.bundle["model"].named_steps["classifier"].classes_
        profile = dict(self.profile, Current_Course=str(labels[0]),
                       Completed_Courses=[str(label) for label in labels[1:]])
        self.assertTrue(recommend(profile, 5, self.bundle).empty)


if __name__ == "__main__":
    unittest.main()
