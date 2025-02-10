#!/usr/bin/env python
# coding: utf-8

# # L4: Predictions, Prompts and Safety

# - Load the Project ID and credentials.

# In[1]:


from utils import authenticate


credentials, PROJECT_ID = authenticate() 


# In[2]:


REGION = 'us-central1'


# - Import the [Vertex AI](https://cloud.google.com/vertex-ai) SDK.
# - Import and load the model.
# - Initialize it.

# In[3]:


import vertexai
from vertexai.language_models import TextGenerationModel


# In[4]:


vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials,
)


# ## Deployment
# 
# ### Load Balancing

# - Load from pre-trained `text-bison@001`
# - Retrieve the endpoints (deployed as REST API)

# In[5]:


model = TextGenerationModel.from_pretrained(
    'text-bison@001'
)


# - Get the list of multiple models executed and deployed.
# - This helps to rout the traffic to different endpoints.

# In[7]:


tuned_models = model.list_tuned_model_names()


# In[8]:


for model in tuned_models:
    print(model)


# - Randomly select from one of the endpoints to divide the prediction load.

# In[9]:


import random


# In[12]:


selected_tuned_model = random.choice(tuned_models)


# ### Getting the Response
# 
# - Load the endpoint of the randomly selected model to be called with a prompt.
# - The prompt needs to be as similar as possible as the one you trained your model on (python questions from stack overflow dataset)

# In[13]:


deployed_model = TextGenerationModel.get_tuned_model(
    selected_tuned_model
)


# In[15]:


PROMPT = 'How can I load a csv file using Pandas?'


# - Use `deployed_model.predict` to call the API using the prompt. 

# In[16]:


# depending on the latency of your prompt
# it can take some time to load
response = deployed_model.predict(PROMPT)


# In[17]:


print(response)


# - `pprint` makes the response easily readable.

# In[18]:


from pprint import pprint


# - Sending multiple prompts can return multiple responses `([0], [1], [2]...)`
# - In this example, only 1 prompt is being sent, and returning only 1 response `([0])`

# In[23]:


output = response._prediction_response[0]


# In[20]:


pprint(output)


# In[24]:


output = output[0]


# In[25]:


pprint(output)


# In[26]:


final_output = output['content']


# In[27]:


print(final_output)


# #### Prompt Management and Templates
# - Remember that the model was trained on data that had an `Instruction` and a `Question` as a `Prompt` (Lesson 2).
# - In the example above, *only*  a `Question` as a `Prompt` was used for a response.
# - It is important for the production data to be the same as the training data. Difference in data can effect the model performance.
# - Add the same `Instruction` as it was used for training data, and combine it with a `Question` to be used as a `Prompt`.

# In[33]:


INSTRUCTION = """
Please answer the following Stackoverflow question on Python.
Answer it like you are a developer answering Stack Overflow questions.
Question:
"""


# In[34]:


QUESTION = """
How can I store my TensorFlow checkpoint on
Google Cloud Storage? Python example?"
"""


# - Combine the intruction and the question to create the prompt.

# In[35]:


PROMPT = f"{INSTRUCTION} {QUESTION}"


# In[36]:


print(PROMPT)


# - Get the response using the new prompt, which is consistent with the prompt used for training.

# In[37]:


final_response = deployed_model.predict(PROMPT)


# In[38]:


output = final_response._prediction_response[0][0]['content']


# - Note how the response changed from earlier.

# In[39]:


print(output)


# ### Safety Attributes
# - The reponse also includes safety scores.
# - These scores can be used to make sure that the LLM's response is within the boundries of the expected behaviour.
# - The first layer for this check, `blocked`, is by the model itself.

# In[41]:


output_old = response._prediction_response[0][0]
blocked = output_old['safetyAttributes']['blocked']


# - Check to see if any response was blocked.
# - It returns `True` if there is, and `False` if there's none.

# In[42]:


print(blocked)


# - The second layer of this check can be defined by you, as a practitioner, according to the thresholds you set.
# - The response returns probabilities for each safety score category which can be used to design the thresholds.

# In[43]:


safety_attributes = output_old['safetyAttributes']


# In[44]:


pprint(safety_attributes)


# ### Citations
# - Ideally, a LLM should generate as much original cotent as possible.
# - The `citationMetadata` can be used to check and reduce the chances of a LLM generating a lot of existing content.

# In[45]:


citation = output_old['citationMetadata']['citations']


# - Returns a list with information if the response is cited, if not, then it retuns an empty list.

# In[46]:


pprint(citation)


# ## Try it Yourself! - Return a Citation
# 
# Now it is time for you to come up with an example, for which the model response should return citation infomration. The idea here is to see how that would look like, so keeping it basic, the prompt can be different than the coding examples used above. Below code is one such prompt:

# In[47]:


PROMPT = 'Finish the sentence: To be, or not '


# In[52]:


response = deployed_model.predict(PROMPT)


# In[53]:


output = response._prediction_response[0][0]
print(output['content'])


# In[54]:


citation = output['citationMetadata']['citations']
pprint(citation)


# **Your turn!**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Optional Notebook
# 
# [Tuning and deploy a foundation model](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/tuning/tuning_text_bison.ipynb)

# In[ ]:




