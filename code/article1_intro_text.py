"""Patch Article 1 DOCX: fill humanised Introduction; keep Methods/Results/figures."""

from __future__ import annotations

# Rebuild via existing builder after patching intro block in build_article1_docx.py
# This file only stores the introduction text used by the builder.

INTRO_PARAS = [
    # opening
    (
        "Sport tourism sits in an awkward but productive place between leisure travel, "
        "event organisation and destination development. For host regions it is rarely "
        "just another visitor segment: events, active travel and sport-related mobility "
        "reshape seasonal demand, place branding and local business opportunity at the "
        "same time (Higham & Hinch, 2018; Getz & Page, 2016). Over the last two decades "
        "the scholarly conversation has widened well beyond spectator flows. Recent "
        "bibliometric work shows a field that now routinely mixes tourist behaviour, "
        "event imaging, sustainability and destination development, with a visible "
        "acceleration of writing since the early 2000s (Arici et al., 2025). What is "
        "still less settled is how that literature talks about *innovation* and "
        "*entrepreneurship* when sport tourism is the empirical setting rather than a "
        "passing keyword."
    ),
    (
        "That omission matters because practice has moved faster than the map of "
        "concepts. Destination agencies and event organisers increasingly treat digital "
        "platforms, new venture formats and green operating models as ordinary tools of "
        "competition, not as exotic add-ons (Ratten, 2018; González-Serrano et al., 2020). "
        "Smart technologies are being woven through event stages for experience design, "
        "marketing and sustainability reporting (Thelen & Kim, 2024), "
        "while carbon, logistics and stakeholder disclosure have become harder to ignore "
        "in major sport calendars. In parallel, smaller and medium-sized events continue "
        "to be defended as more place-sensitive engines of local economic and social "
        "value (Rossini et al., 2024). Put simply, innovation in this domain is no longer "
        "only about a new race format or a better ticket app; it stretches across "
        "markets, technologies, communities and host-city politics."
    ),
    (
        "Entrepreneurship research has followed a similar widening. Sport entrepreneurship "
        "is now discussed as organisational experimentation, lifestyle venturing and "
        "socially oriented business models, not merely start-up creation inside clubs "
        "(Ratten, 2012; 2018). In sport-tourism settings specifically, studies of "
        "entrepreneurial behaviour in developing destinations and of lifestyle "
        "entrepreneurship around surf and outdoor travel point to opportunity structures "
        "that look quite different from those of conventional hospitality SMEs "
        "(Heydari et al., 2022; Ivanycheva et al., 2024). Yet much of this work remains "
        "case-based or thematically fragmented. We know pieces of the puzzle—event "
        "legacies, tourist motivation, digital fan engagement, sustainable hosting—but "
        "we still lack a clear picture of how these pieces sit next to one another in "
        "the published knowledge base."
    ),
    (
        "Existing reviews help, though not enough for the question we pose here. "
        "Broad sport-tourism bibliometrics have clarified behavioural, event-image and "
        "sustainability clusters (Arici et al., 2025). Work on sustainable sport "
        "entrepreneurship and innovation has already flagged tourism as an under-developed "
        "bridge inside that literature and called for tighter coupling of entrepreneurship, "
        "environment and innovation agendas (González-Serrano et al., 2020). Sport "
        "management mapping likewise shows a maturing but specialised field in which "
        "innovation themes do not automatically align with tourism debates "
        "(Hammerschmidt et al., 2024). What these studies seldom do is isolate the "
        "intersection of sport tourism/events with innovation *and* entrepreneurship, "
        "then examine both the latent topics and the relational bridges that hold those "
        "topics together. Citation or co-word maps of the wider field are useful; they "
        "are not a substitute for a focused conceptual anatomy of this particular niche."
    ),
    (
        "There is also a methodological reason to reopen the question. Science-mapping "
        "guidance now treats bibliometric text as a legitimate knowledge base for "
        "recovering field structure, provided cleaning, transparency and interpretation "
        "are taken seriously (Donthu et al., 2021; Aria & Cuccurullo, 2017). Classic "
        "co-word analysis already argued that concepts gain meaning through their "
        "networks of association (Callon et al., 1983). More recent practice combines "
        "topic modelling with network metrics so that themes can be read alongside the "
        "terms that stitch themes together (van Eck & Waltman, 2014). For a fast-moving, "
        "multi-outlet literature—sport management, tourism, events, sustainability—"
        "that combination is attractive. It lets us ask not only “what are people writing "
        "about?” but also “which ideas do the brokerage work?”"
    ),
    (
        "Against that backdrop, this paper examines the conceptual structure of "
        "innovation and entrepreneurship research in sport tourism and sport-event "
        "contexts. Using a cleaned Scopus corpus and a pipeline that joins TF–IDF "
        "salience, non-negative matrix factorisation (NMF) topics, and Jaccard-weighted "
        "concept networks with Louvain communities and betweenness bridges, we address "
        "four research questions:"
    ),
]

RQ_LIST = [
    "**RQ1.** What thematic clusters organise innovation and entrepreneurship scholarship in sport tourism and sport-event settings?",
    "**RQ2.** How are innovation, entrepreneurship, digital technology and destination development linked in the concept network?",
    "**RQ3.** Which bridging concepts connect markets, events and tourism systems?",
    "**RQ4.** How has the field’s vocabulary shifted since 2019, particularly around sustainability and digitalisation?",
]

INTRO_CLOSING = [
    (
        "The intended contribution is threefold. Empirically, we offer a focused map of "
        "a niche that general sport-tourism reviews only brush against. Conceptually, we "
        "distinguish adjacent but non-identical positions—destination-market systems, "
        "entrepreneurial event ventures, digital–fan infrastructures, green transition "
        "discourses and Olympic/mega-event hosting—and identify the managerial and "
        "modelling terms that appear to hold them in conversation. Methodologically, we "
        "show how topic modelling and concept-network bridges can be read together, "
        "rather than treated as alternative visualisations of the same list of keywords. "
        "The remainder of the paper reviews the relevant literatures, details the "
        "corpus and analytical strategy, reports the findings, and discusses implications "
        "for theory and for destination and event practice."
    ),
]
