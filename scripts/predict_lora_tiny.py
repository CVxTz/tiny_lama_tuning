from peft import AutoPeftModelForCausalLM
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path
from tiny_lama_tuning.generation_utils import get_logit_criteria


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parents[1] / "outputs"

    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    # llama_tokenizer.pad_token = llama_tokenizer.eos_token
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=quant_config,
        device_map={"": 0},
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        refined_model, device_map={"": 0}
    ).merge_and_unload()
    model.eval()

    text_gen = pipeline(
        task="text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
    )
    text_gen.model = model

    _context = "Obama associate tells liberals to sabotage Trump's election chances at the ballot box"
    _context = "Clinton's 'deplorables' slip: 2012 campaign hints it's not a game-changer Hillary Clinton ’ s weekend comment that “ half ” of Donald Trump ’ s supporters are a racist and sexist “ basket of deplorables ” is still roiling the presidential race as the workweek begins . Lots of pundits are comparing it to Mitt Romney ’ s famous “ 47 percent ” statement , in which he said nearly half of American voters can be written off as welfare moochers .Lost in most of the discussion of this comparison is the fact that Romney ’ s “ 47 percent ” words , revealed when a secret source leaked the tape of a fundraiser to Mother Jones magazine , didn ’ t much affect the 2012 outcome , and probably did not even move the polls that much .That ’ s the political reality behind such moments as Mrs. Clinton ’ s “ deplorables ” or Romney ’ s “ 47 percent ” : Voters usually don ’ t make up their minds from a few days of news coverage .It ’ s true that many voters saw Romney ’ s perceived gaffe in negative terms . And it sure seemed like something that would have serious negative repercussions : harsh words , seemingly delivered in secret , about an opponent ’ s supporters . That made it seem more important than a typical political slip of the tongue .But in terms of who people planned to vote for , “ there was no consistent evidence that much changed ” in the wake of the tape ’ s release , wrote political scientists John Sides of George Washington University and Lynn Vavreck of UCLA in their history of the 2012 election , “ The Gamble . ”Gallup poll data showed President Obama ’ s lead over Romney actually shrank from 3 to 2 percentage points the week after “ 47 percent ” became public . Rasmussen polls stayed the same . The average of all public polls was “ stable ” in the wake of the controversy , according to Sides and Vavreck .In other words , peoples ’ opinions about the race did not really alter , on either side .In contrast , the first 2012 debate , held on Oct. 3 , did move the polls . The media roundly declared Romney the victor over a flat Obama . Some surveys even put Romney in the lead . ( Spoiler alert : He lost . The fallout of subsequent debates reversed those gains . )What ’ s the takeaway from this ? Maybe that lots of things the media says are game-changers , aren ’ t .Get the Monitor Stories you care about delivered to your inbox . By signing up , you agree to our Privacy PolicyIt ’ s certainly possible that Clinton ’ s “ deplorables ” comment could hurt her . Insulting ordinary voters is not something campaign consultants generally urge . But in general presidential races are not unstable . Leads shrink or widen slowly , driven by fundamentals such as the state of the economy , or predictable dynamics such as Republican voters rallying around Trump . At this point in the cycle , many people ’ s minds are set , and it takes a lot to change them .There are break points , but they tend to be set news events that draw massive coverage . The conventions are one – Clinton jumped out to a big lead following the close of the Democratic National Convention . The debates might be another . Thus the first direct clash between Clinton and Trump , set for Sept. 26 , is likely to be more consequential than Clinton ’ s insult ."
    prompt = f"What is the political bias of this new article?\nContext: {_context}\nAnswer: "

    print(prompt)

    outputs = text_gen(
        prompt.strip(),
        do_sample=False,
        return_full_text=False,
        max_new_tokens=200,
        # logits_processor=get_logit_criteria(tokenizer=llama_tokenizer),
    )

    print(outputs[0]["generated_text"])
