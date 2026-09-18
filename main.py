from agent.graph import build_graph

if __name__ == '__main__':
    graph = build_graph()
    print('Pipeline initialized')
    
    input_data = {

        "product_image":
            "data/input/serum.jpeg",

        "product_description":
            """
            Premium vitamin C serum
            for luxury skincare brand
            """
    }


    result = graph.invoke(
        input_data
    )


    print("\nFINAL RESULT")
    print(result)
