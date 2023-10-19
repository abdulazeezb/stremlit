# Styles related to headers and generic text alignment
header_style = """
<style>
    h1 {
        text-align: center;
    }
</style>
"""

# Styles for text formatting and common design elements
text_style = """
<style>
    /* Common font settings */
    .big-font {
        font-weight: bold;
        font-size: 16px;
        font-family: sans-serif;
        color: #e0e0e0;  /* Light gray */
    }

    /* Card settings */
    .custom-card {
        background-color: #333333;  /* Dark gray */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
    }

    /* List settings inside card */
    .custom-card ul {
        list-style: disc inside;  /* Marker style and position */
        padding-left: 20px;  /* Indentation */
    }
    .custom-card li {
        margin-bottom: 10px;  /* Spacing between items */
    }
</style>
"""

# Styles for metric boxes and their content
metric_style = """
<style>
    /* Generic styles for metrics */
    .header, .value {
        font-weight: bold;
        color: #E0E0E0;
    }

    /* Styling for the container of metrics */
    .metric {
        border: 1px solid #FFFFFF;
        background-color: #1A1A1A;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        height: 90px;
        text-align: center;
    }

    /* Text formatting within the metric boxes */
    .header {
        font-size: 1.2em;
    }
    .value {
        font-size: 1.5em;
    }
    .change {
        font-size: 1em;
        color: #FF4500;  /* Red-orange for emphasis */
    }

    /* Additional layout adjustments */
    #pushed-content {
        margin-top: 75px;  /* To avoid overlapping with other content */
    }
</style>
"""
