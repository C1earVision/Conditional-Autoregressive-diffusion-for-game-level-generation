from data.parser import LevelParser
from data.extractor import PatchExtractor
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
import numpy as np
import pickle

import glob

parser = LevelParser()
extractor = PatchExtractor()
patch_evaluator = PatchDifficultyEvaluator(parser)

rawData = []
for filepath in glob.glob("./dataset/*.txt"):
    with open(filepath, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
        rawData.append(content)

parsed_dataset = parser.parse_dataset(
    levels=rawData,
    level_names=[f"map_{i+1}" for i in range(1000)]
)


patches, metadata = extractor.extract_patches_from_dataset(
parsed_dataset
)


patch_evaluation_results = patch_evaluator.evaluate_patches_batch(patches, metadata)
for i, result in enumerate(patch_evaluation_results):
  metadata[i]['final_score'] = result['scores']['difficulty_score']

np.save('./output/processed/patches.npy', patches)
with open('./output/processed/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)