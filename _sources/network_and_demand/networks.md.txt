---
title: Networks & Demand
firstpage:
---

# Networks

Some of the networks utilized in this study include **Ingolstadt**, **Cologne**, **Csomor**, and **Manhattan**, each representing different traffic scenarios for evaluating the impact of autonomous vehicles (AVs) on urban mobility.

For a detailed definition of the network names and their corresponding keychain:  
[**Network Keychain**](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/9c02a6eca4df0900460ab44d4029a77eeb10e5ff/routerl/keychain.py#L121C5-L121C135).

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 48%;">
        <img src="../_static/ingolstadt.png" alt="Ingolstadt" width="100%" />
        <figcaption style="text-align: center;">Ingolstadt Network</figcaption>
    </figure>
    <figure style="width: 48%;">
        <img src="../_static/cologne.png" alt="Cologne" width="100%" />
        <figcaption style="text-align: center;">Cologne Network</figcaption>
    </figure>
</div>

<br>
<br>

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 48%;">
        <img src="../_static/csomor.png" alt="Csomor" width="100%" />
        <figcaption style="text-align: center;">Csomor Network</figcaption>
    </figure>
    <figure style="width: 48%;">
        <img src="../_static/two_route_yield.png" alt="Two Route Network" width="100%" />
        <figcaption style="text-align: center;">Two Route Network</figcaption>
    </figure>
</div>

<br>
<br>

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 48%;">
        <img src="../_static/arterial.png" alt="Arterial" width="100%" />
        <figcaption style="text-align: center;">Arterial Network</figcaption>
    </figure>
    <figure style="width: 48%;">
        <img src="../_static/grid.png" alt="Grid" width="100%" />
        <figcaption style="text-align: center;">Grid Network</figcaption>
    </figure>
</div>

<br>
<br>

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 48%;">
        <img src="../_static/ortuzar.png" alt="Ortuzar" width="100%" />
        <figcaption style="text-align: center;">Ortuzar Network</figcaption>
    </figure>
    <figure style="width: 48%;">
        <img src="../_static/manhattan.png" alt="Manhattan" width="100%" />
        <figcaption style="text-align: center;">Manhattan Network</figcaption>
    </figure>
</div>

<br>
<br>

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 48%;">
        <img src="../_static/nguyen.png" alt="Nguyen" width="100%" />
        <figcaption style="text-align: center;">Nguyen Network</figcaption>
    </figure>
    <figure style="width: 48%;">
        <img src="../_static/grid6.png" alt="Grid 6" width="100%" />
        <figcaption style="text-align: center;">Grid 6 Network</figcaption>
    </figure>
</div>

# Demand

The [`agent_generation.py`](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/routerl/environment/agent_generation.py) file defines the network demand for each experiment by specifying the number of agents, their origins, destinations, and start times.