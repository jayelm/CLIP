import os


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
.pos_img_div, .neg_img_div {{
    display: inline-block;
    width: 100px;
}}
.correct {{
    background-color: #e5ffe5 !important;
}}
.incorrect {{
    background-color: #ffe5e5 !important;
}}
img {{
    width: 100px;
    height: 100px;
}}
.score, .pos-indicator, .neg-indicator {{
    display: block;
    width: 100px;
    text-align: center;
}}
</style>
<title>Results</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
</head>
<body>
<h1 class="title">{title}</h1>
{contents}
</body>
</html>
"""


EX_TEMPLATE = """
<div class="ex card {background}">
<div class="card-body">
    <h6 class="card-subtitle mb-2 text-muted">{text}</h6>
    <div class="pos_img_div">
        <span class="pos-indicator">pos</span>
        <img class="pos_img" src="{pos_img_url}">
        <span class="score">{pos_img_score:.2f}</span>
    </div>
    <div class="neg_img_div">
        <span class="neg-indicator">neg</span>
        <img class="neg_img" src="{neg_img_url}">
        <span class="score">{neg_img_score:.2f}</span>
    </div>
</div>
</div>
"""


def vis(data, acc, dname):
    result_dir = f"{dname}_result"
    result_fname = os.path.join(result_dir, "index.html")
    os.makedirs(result_dir, exist_ok=True)

    exs = []

    for i, (pi, ni, txt, scores, correct) in enumerate(data):
        # Save images
        pi_name = f"{i}_pos.jpg"
        pi.save(os.path.join(result_dir, pi_name))

        ni_name = f"{i}_neg.jpg"
        ni.save(os.path.join(result_dir, ni_name))

        pos_img_score = scores[0]
        neg_img_score = scores[1]
        exs.append(EX_TEMPLATE.format(
            text=txt.replace(" .", ""),
            pos_img_url=pi_name,
            neg_img_url=ni_name,
            pos_img_score=pos_img_score,
            neg_img_score=neg_img_score,
            background="correct" if correct else "incorrect",
        ))

    html = HTML_TEMPLATE.format(
        title=f"Overall accuracy: {acc * 100:.1f}%",
        contents=''.join(exs),
    )
    with open(result_fname, 'w') as fout:
        fout.write(html)
