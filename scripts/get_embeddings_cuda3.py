import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle as pk
from tqdm import tqdm
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def get_embeddings(model, texts, keys=None):
    """Extracts Text Embeddings
    Args:
        texts (list): List of texts
    Returns:
        dict: Text and corresponding embedding
    """
    if keys is None:
        keys = texts
    embeddings = model.encode(
        texts, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # normalise embeddings
    return dict(zip(keys, embeddings))


if __name__ == "__main__":
    data = pd.read_csv("../csvs/data_humans_allresponses.csv")

    texts_vf = data[data["task"] == 1]["response"].unique().tolist()

    texts = [texts_vf]
    tasks = ["vf"]

    # Note: #words allowed = #tokens/2
    models = ["dunzhang/stella_en_400M_v5"]
    model_to_id = dict(zip(models, ["stella"]))

    # wikipedia_contexts = pk.load(open("../pickle/wikipedia_contexts.pk", "rb"))
    wikipedia_contexts = {
            "culture": "Culture is a concept that encompasses the social behavior, institutions, and norms found in human societies, as well as the knowledge, beliefs, arts, laws, customs, capabilities, attitude, and habits of the individuals in these groups.[1] Culture is often originated from or attributed to a specific region or location. Humans acquire culture through the learning processes of enculturation and socialization, which is shown by the diversity of cultures across societies. A cultural norm codifies acceptable conduct in society; it serves as a guideline for behavior, dress, language, and demeanor in a situation, which serves as a template for expectations in a social group. Accepting only a monoculture in a social group can bear risks, just as a single species can wither in the face of environmental change, for lack of functional responses to the change.[2] Thus in military culture, valor is counted a typical behavior for an individual and duty, honor, and loyalty to the social group are counted as virtues or functional responses in the continuum of conflict. In the practice of religion, analogous attributes can be identified in a social group. Cultural change, or repositioning, is the reconstruction of a cultural concept of a society.[3] Cultures are internally affected by both forces encouraging change and forces resisting change. Cultures are externally affected via contact between societies. Organizations like UNESCO attempt to preserve culture and cultural heritage.",
            "geography": "Geography (from Ancient Greek γεωγραφία geōgraphía; combining gê 'Earth' and gráphō 'write') is the study of the lands, features, inhabitants, and phenomena of Earth.[1] Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science.[2] Geography has been called 'a bridge between natural science and social science disciplines.'[3] Origins of many of the concepts in geography can be traced to Greek Eratosthenes of Cyrene, who may have coined the term 'geographia' (c. 276 BC – c. 195/194 BC).[4] The first recorded use of the word γεωγραφία was as the title of a book by Greek scholar Claudius Ptolemy (100 – 170 AD).[1] This work created the so-called 'Ptolemaic tradition' of geography, which included 'Ptolemaic cartographic theory.'[5] However, the concepts of geography (such as cartography) date back to the earliest attempts to understand the world spatially, with the earliest example of an attempted world map dating to the 9th century BCE in ancient Babylon.[6] The history of geography as a discipline spans cultures and millennia, being independently developed by multiple groups, and cross-pollinated by trade between these groups. The core concepts of geography consistent between all approaches are a focus on space, place, time, and scale.[7][8][9][10][11][12] Today, geography is an extremely broad discipline with multiple approaches and modalities. There have been multiple attempts to organize the discipline, including the four traditions of geography, and into branches.[13][3][14] Techniques employed can generally be broken down into quantitative[15] and qualitative[16] approaches, with many studies taking mixed-methods approaches.[17] Common techniques include cartography, remote sensing, interviews, and surveying.",
            "health": "Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain (including mental pain), or injury. Health can be promoted by encouraging healthful activities, such as regular physical exercise and adequate sleep,[1] and by reducing or avoiding unhealthful activities or situations, such as smoking or excessive stress. Some factors affecting health are due to individual choices, such as whether to engage in a high-risk behavior, while others are due to structural causes, such as whether the society is arranged in a way that makes it easier or harder for people to get necessary healthcare services. Still, other factors are beyond both individual and group choices, such as genetic disorders.",
            "human_behaviour": "Human behavior is the potential and expressed capacity (mentally, physically, and socially) of human individuals or groups to respond to internal and external stimuli throughout their life. Behavior is driven by genetic and environmental factors that affect an individual. Behavior is also driven, in part, by thoughts and feelings, which provide insight into individual psyche, revealing such things as attitudes and values. Human behavior is shaped by psychological traits, as personality types vary from person to person, producing different actions and behavior. Social behavior accounts for actions directed at others. It is concerned with the considerable influence of social interaction and culture, as well as ethics, interpersonal relationships, politics, and conflict. Some behaviors are common while others are unusual. The acceptability of behavior depends upon social norms and is regulated by various means of social control. Social norms also condition behavior, whereby humans are pressured into following certain rules and displaying certain behaviors that are deemed acceptable or unacceptable depending on the given society or culture. Cognitive behavior accounts for actions of obtaining and using knowledge. It is concerned with how information is learned and passed on, as well as creative application of knowledge and personal beliefs such as religion. Physiological behavior accounts for actions to maintain the body. It is concerned with basic bodily functions as well as measures taken to maintain health. Economic behavior accounts for actions regarding the development, organization, and use of materials as well as other forms of work. Ecological behavior accounts for actions involving the ecosystem. It is concerned with how humans interact with other organisms and how the environment shapes human behavior.",
            "mathematics": "Mathematics is a field of study that discovers and organizes methods, theories and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory (the study of numbers), algebra (the study of formulas and related structures), geometry (the study of shapes and spaces that contain them), analysis (the study of continuous changes), and set theory (presently used as a foundation for all mathematics). Mathematics involves the description and manipulation of abstract objects that consist of either abstractions from nature or—in modern mathematics—purely abstract entities that are stipulated to have certain properties, called axioms. Mathematics uses pure reason to prove properties of objects, a proof consisting of a succession of applications of deductive rules to already established results. These results include previously proved theorems, axioms, and—in case of abstraction from nature—some basic properties that are considered true starting points of the theory under consideration.[1] Mathematics is essential in the natural sciences, engineering, medicine, finance, computer science, and the social sciences. Although mathematics is extensively used for modeling phenomena, the fundamental truths of mathematics are independent of any scientific experimentation. Some areas of mathematics, such as statistics and game theory, are developed in close correlation with their applications and are often grouped under applied mathematics. Other areas are developed independently from any application (and are therefore called pure mathematics) but often later find practical applications.[2][3] Historically, the concept of a proof and its associated mathematical rigour first appeared in Greek mathematics, most notably in Euclid's Elements.[4] Since its beginning, mathematics was primarily divided into geometry and arithmetic (the manipulation of natural numbers and fractions), until the 16th and 17th centuries, when algebra[a] and infinitesimal calculus were introduced as new fields. Since then, the interaction between mathematical innovations and scientific discoveries has led to a correlated increase in the development of both.[5] At the end of the 19th century, the foundational crisis of mathematics led to the systematization of the axiomatic method,[6] which heralded a dramatic increase in the number of mathematical areas and their fields of application. The contemporary Mathematics Subject Classification lists more than sixty first-level areas of mathematics."
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for modelname in models:
        print(modelname)
        model = SentenceTransformer(f"../../models/{modelname[modelname.find("/") + 1:]}", trust_remote_code=True)
        model = model.to(device)

        for i, textset in enumerate(texts):
            # contexts_ofrelevance = wikipedia_contexts[tasks[i]]
            contexts_ofrelevance = wikipedia_contexts

            for context in contexts_ofrelevance:
                print(context)
                
                textset_contextual = [
                    contexts_ofrelevance[context] + " " + text for text in textset
                ]
                embeddings_dict = get_embeddings(model, textset_contextual, textset)
                
                context = context.replace("(", "")
                context = context.replace(")", "")
                context = context.replace("_", "")
                context = context.replace("/", "")
                with open(
                    f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}_{context}.pk",
                    "wb",
                ) as f:
                    pk.dump(embeddings_dict, f)
