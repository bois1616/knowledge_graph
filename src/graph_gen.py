# -*- coding: UTF-8 -*-

# import

# globals

# functions


if __name__ == '__main__':
    # Function to generate the network diagram using D3.js library
    def generate_network(data)
        # Required libraries
        require
        'json'
        require
        'csv'

        # D3.js library for generating visualizations
        require
        'd3'

        # Define the dimensions of the chart
        width = 960
        height = 500

        # Load the JSON data and parse it
        json_data = JSON.parse(File.open("data.json", "r"))
        links = []
        nodes = {}

        # Generate unique node IDs for each term
        json_data.each
        do | item |
        nodes[item["term"]] | |= {id: SecureRandom.hex(6), label: item["term"], x: rand(width), y: rand(height)}


    end

    # Generate links between related terms
    json_data.each
    do | source, targets |
    next
    unless
    nodes[source] & & !targets.empty?

    targets.each
    do | target |
    link = {source: nodes[source][:id], target: nodes[target][:id], value: rand(10)}
    links << link
end
end

# Define the layout of the network
layout = D3::Layout::Force.new
do | simulation |
simulation.size([width, height])
simulation.links(links)
simulation.nodes(nodes.values)
simulation.link_distance(50)
simulation.gravity(0.1)
simulation.center([width / 2, height / 2])
simulation.start_padding(1)
simulation.stop_padding(1)
simulation.velocity_decay(0.9)
simulation.update
end

# Define the SVG canvas to render the chart on
svg = D3::Select.new("#network").append("svg")
do | element |
element.attr("viewBox", "0 0 #{width} #{height}")
element.style("background-color", "#f9f9f9")
end

# Define the circles for each node in the network
circles = svg.select_all(".circle").data(nodes.values)
do | selection |
selection.enter.append("circle")
do | element |
element.attr("class", "circle")
element.attr("r", 12)
element.attr("fill", lambda {nodes[d["label"]] & & nodes[d["label"]][: color] | | "#f7dc6f"})
end
end

# Add labels for each node
svg.select_all(".text").data(nodes.values)
do | selection |
selection.enter.append("text")
do | element |
element.attr("class", "text")
element.attr("x", lambda {d["x"]})
                         element.attr("y", lambda {d["y"] - 20})
                         element.attr("fill", lambda {nodes[d["label"]] & & nodes[d["label"]][: color] | | "#f7dc6f"})
end
end

# Add arrows to the links between related terms
svg.select_all(".link").data(links)
do | selection |
selection.enter.append("line")
do | element |
element.attr("class", "link")
element.attr("stroke-width", lambda {rand(2..4)})
                                    element.attr("stroke", "#ddd")
                                    end

                                    # Define the arrowhead for each link
                                    element.append("svg:polyline") do | arrow |
                                    arrow.attr("points", "0, 6 : 6, 6 : 6, 0")
                                    arrow.attr("stroke-width", lambda {rand(2..4)})
                                    arrow.attr("fill", "none")
                                    arrow.attr("stroke", "#ddd")
                                    end

                                    # Define the text for each link
                                    element.append("text") do | label |
                                    label.attr("class", "link_label")
                                    label.attr("x", lambda {rand(-10..10)})
                                    label.attr("y", lambda {rand(8..12)})
                                    label.attr("fill", "#fff")
                                    end
                                    end

                                    # Render the chart
                                    layout.run(500)

                                    # Define the tooltip for each node
                                    def show_tooltip(d)
                                    content = d["label"] + "\n" + d["definition"]
                                    D3:: Select("#tooltip").style("display", "inline")
D3::Select("#tooltip .content").html(content)
D3::Select("#tooltip .arrow:first-child").attr("transform", "translate(10px," + (d["y"] - 50).to_s + ") rotate(-45)")
D3::Select("#tooltip .arrow:last-child").attr("transform",
                                              "translate(" + (-20).to_s + "," + (-70).to_s + ") rotate(45)"
                                              )
end

# Define the tooltip that appears when hovering over a node
D3::Select(".circle")
do | selection |
selection.on("mouseover", lambda {show_tooltip(d)})
                                 selection.on("mouseout", lambda {hide_tooltip()})
                                 end

                                 # Define the function to hide the tooltip
                                 def hide_tooltip()
                                 D3:: Select("#tooltip").style("display", "none")
D3::Select("#tooltip .arrow:first-child").attr("transform", "translate(10px,0) rotate(-45)")
D3::Select("#tooltip .arrow:last-child").attr("transform", "translate(0,-20) rotate(45)")
end

# Render the tooltip in a separate element
html = << HTML
< div
id = "tooltip" >
< div


class ="content" > < / div >


< svg
viewBox = "0 0 100 20"
style = "display:inline-block; width:0px; height:0px; vertical-align:middle;" >
< polyline
points = "0,10 5,15 10,10 15,15"


class ="arrow" > < / polyline >


< polyline
points = "85,10 80,15 75,10 70,15"


class ="arrow" > < / polyline >


< / svg >
< / div >
HTML
CSV.open("tooltip.csv", "w")
do | csv |
csv << %{label, definition}
json_data.each
do | term, synonyms |
csv << [term, synonyms.join(", ") if synonyms]
end
end
end

# Load the JSON data and generate the network diagram
generate_network(json_data)

# Define the CSS styles for the chart
CSS = "
.circle
{
    cursor: pointer;
}

.text
{
    font - size: 12px;
pointer - events: none;
}

.link
{
    stroke:  # ddd;
        stroke - opacity: 0.6;
}

.link_label
{
    font - size: 12px;
fill: white;
text - anchor: middle;
pointer - events: none;
}

# tooltip {
display: none;
position: absolute;
z - index: 1000;
padding: 8
px;
background - color:  # f9f9f9;
border: 1
px
solid  # ddd;
border - radius: 4
px;
box - shadow: 2
px
2
px
3
px
rgba(0, 0, 0, 0.3);
}

# tooltip .content {
margin: 6
px;
line - height: 18
px;
}

# tooltip .arrow {
stroke:  # ddd;
stroke - width: 1.5
px;
}
"

# Add the CSS styles to the <head> section of the HTML file
open("assets/styles.css", "a")
do | file |
file.write(CSS)
end

puts
"Network diagram generated successfully!"

