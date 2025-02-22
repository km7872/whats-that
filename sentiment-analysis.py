from mediapipe.tasks import python
from mediapipe.tasks.python.text import TextClassifierOptions
from mediapipe.tasks.python import BaseOptions

model_path = 'model_path' # model goes here
input_text = 'input_text' # text goes here

# Initialize BaseOptions correctly
base_options = BaseOptions(model_asset_path=model_path)

# Initialize TextClassifierOptions with the correct base_options
options = TextClassifierOptions(base_options=base_options)

# Use the TextClassifier to classify the input text
with python.text.TextClassifier.create_from_options(options) as classifier:
    classification_result = classifier.classify(input_text)

if classification_result and classification_result.classifications:
  for classification in classification_result.classifications[0].categories:
      print(f"Category: {classification.category_name}, Score: {classification.score}")
