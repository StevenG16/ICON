import matplotlib.pyplot as plt



def plot_factors(factors, img_name = "test"):

    yaxis =  [i for i in range(16)]
    xaxis = factors
    factors_labels = ["Warmth", "Reasoning", "Emotional stability", "Dominance",
                        "Liveliness", "Rule-consciousness", "Social boldness", "Sensitivity",
                        "Vigilance", "Abstractedness", "Privateness", "Apprehension",
                        "Openness to change", "Self-reliance", "Perfectionism", "Tension"
                    ]

    plt.rcParams["figure.figsize"] = (18, 10.5)  
    fig, axis = plt.subplots()
    axis.plot(xaxis, yaxis, marker = "o", markersize = 10)
    axis.set_yticks([x for x in range(16)], factors_labels)
    axis.set_xlim(left = 0.0, right = 1.0)
    axis.set_xticks([x/10.0 for x in range(11)])
    axis.tick_params(labelsize=15)
    axis.grid(True, which="both")
    axis.margins(y=0.07)

      
    plt.savefig(f"imgs/{img_name}.png") 


def plot_elbow(num_clusters, inertias):
    plt.plot(range(1, num_clusters+1), inertias)
    plt.savefig(f"imgs/elbow.png")
