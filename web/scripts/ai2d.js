document.getElementById('imageInput').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Choose Image';
    document.getElementById('fileLabel').innerHTML = `<i class="fa-solid fa-image"></i> ${fileName}`;
});

let currentJsonData = null;

function downloadJSON() {
    if (!currentJsonData) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentJsonData, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "diagram_analysis.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

function buildExactGraph(containerId, jsonData) {
    if (!jsonData || !jsonData.blobs || !jsonData.relationships) return;

    const nodesArray = [];
    const edgesArray = [];

    const colors = {
        root: "#FFD700",
        super_node: "#FF4500",
        blob: "#90EE90",
        text: "#87CEEB"
    };

    const imageId = jsonData.image_id ? jsonData.image_id.split('.')[0] : "Root";
    const rootId = "Image_Root";
    nodesArray.push({ 
        id: rootId, 
        label: `I${imageId}`, 
        color: { background: colors.root, border: '#333' }, 
        shape: 'ellipse',
        font: { face: 'bold' }
    });

    const processedBlobs = new Set();
    const processedTexts = new Set();
    const idToSuper = {};

    const blobMatches = jsonData.relationships.filter(r => r['relation_type:'] === 'blob_label');
    
    blobMatches.forEach(rel => {
        const t_id = rel.source; 
        const b_id = rel.target;
        const super_id = `${b_id}+${t_id}`;

        idToSuper[b_id] = super_id;
        idToSuper[t_id] = super_id;
        processedBlobs.add(b_id);
        processedTexts.add(t_id);

        nodesArray.push({ 
            id: super_id, 
            label: super_id, 
            color: { background: colors.super_node, border: '#333' }, 
            shape: 'ellipse' 
        });

        edgesArray.push({ from: rootId, to: super_id, label: "has a", arrows: "to", color: "gray" });

        edgesArray.push({ from: super_id, to: b_id, color: "lightgray" });
        edgesArray.push({ from: super_id, to: t_id, color: "lightgray" });
    });

    jsonData.blobs.forEach(blob => {
        nodesArray.push({ 
            id: blob.id, 
            label: blob.id, 
            color: { background: colors.blob, border: '#333' }, 
            shape: 'ellipse' 
        });
        
        if (!processedBlobs.has(blob.id)) {
            edgesArray.push({ from: rootId, to: blob.id, label: "has a", arrows: "to", color: "gray" });
        }
    });

    jsonData.texts.forEach(t => {
        nodesArray.push({ 
            id: t.id, 
            label: t.id, 
            color: { background: colors.text, border: '#333' }, 
            shape: 'ellipse' 
        });

        if (!processedTexts.has(t.id)) {
            edgesArray.push({ from: rootId, to: t.id, label: "has a", arrows: "to", color: "gray" });
        }
    });

    const mainRelationships = jsonData.relationships.filter(r => r['relation_type:'] !== 'blob_label');
    
    mainRelationships.forEach(rel => {
        let srcNode = rel.source;
        let tgtNode = rel.target;

        if (idToSuper[srcNode]) srcNode = idToSuper[srcNode];
        if (idToSuper[tgtNode]) tgtNode = idToSuper[tgtNode];

        edgesArray.push({
            from: srcNode,
            to: tgtNode,
            label: rel['relation_type:'],
            arrows: "to",
            color: "gray",
            font: { align: 'middle' }
        });
    });

    const container = document.getElementById(containerId);
    const graphData = {
        nodes: new vis.DataSet(nodesArray),
        edges: new vis.DataSet(edgesArray)
    };
    
    const options = {
        physics: {
            enabled: true,
            barnesHut: {
                gravitationalConstant: -3000,
                centralGravity: 0.1,
                springLength: 150,
                springConstant: 0.04
            }
        },
        layout: { randomSeed: 42 } 
    };

    new vis.Network(container, graphData, options);
}

async function processDiagram() {
    const fileInput = document.getElementById('imageInput');
    const statusText = document.getElementById('status');
    const gallery = document.getElementById('image-gallery');
    
    if (fileInput.files.length === 0) {
        alert("Please select an image first!");
        return;
    }

    gallery.innerHTML = ""; 
    statusText.innerHTML = `
            <div class="loading-container">
                <img src="imgs/loading.gif" alt="Loading..." class="loading-gif">
                <span class="loading-text">Analyzing image... (this might take a minute)</span>
            </div>
        `;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const inputImageUrl = URL.createObjectURL(fileInput.files[0]);

    try {
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Server error");
        
        const data = await response.json();
        currentJsonData = data.json_data; 
        statusText.innerText = "Analysis Complete!";

        const orderedCards = [
            { title: "Input Image", url: inputImageUrl },
            { title: "Relationships", url: data.images?.relationships },
            { title: "Text Matching", url: data.images?.text_matching },
            { title: "CLIP Results", url: data.images?.clip_results },
            { title: "Knowledge Graph", url: data.images?.knowledge_graph } 
        ];

        orderedCards.forEach(card => {
            if (card.url) {
                const cardDiv = document.createElement('div');
                cardDiv.className = "ai2d-result-card";
                cardDiv.innerHTML = `
                    <h4>${card.title}</h4>
                    <a href="${card.url}" target="_blank">
                        <img src="${card.url}" alt="${card.title}">
                    </a>
                `;
                gallery.appendChild(cardDiv);
            }
        });

        const jsonCard = document.createElement('div');
        jsonCard.className = "ai2d-json-card";
        jsonCard.innerHTML = `
            <h4>Final JSON Data</h4>
            <pre id="json-output">${JSON.stringify(currentJsonData, null, 2)}</pre>
            <button class="json-download-btn" onclick="downloadJSON()">
                <i class="fa-solid fa-download"></i> Download JSON
            </button>
        `;
        gallery.appendChild(jsonCard);

        if (currentJsonData && currentJsonData.blobs && currentJsonData.relationships) {
            const graphCard = document.createElement('div');
            graphCard.className = "ai2d-result-card interactive-graph-card";
            graphCard.innerHTML = `
                <h4>Interactable Scene Graph</h4>
                <div id="mynetwork"></div>
            `;
            gallery.appendChild(graphCard);

            buildExactGraph('mynetwork', currentJsonData);
        }
        
    } catch (error) {
        statusText.innerText = "Error connecting to the AI server.";
        console.error(error);
    }
}

const dynamicGallery = document.getElementById('image-gallery');
const modalElement = document.getElementById("zoomModal");
const dynamicZoomedImg = document.getElementById("zoomedImg");
const dynamicZoomedText = document.getElementById("zoomedText");
const modalImgClass = document.querySelector('.modal-content-image');
const modalTextClass = document.querySelector('.modal-content-text');
const dynamicCloseBtn = document.getElementById("closeZoom");

dynamicGallery.addEventListener('click', function(event) {
    if (event.target.tagName === 'IMG' && event.target.closest('.ai2d-result-card')) {
        event.preventDefault(); 
        
        let imgSrc = event.target.src;
        modalElement.style.display = "block";
        dynamicZoomedImg.src = imgSrc;
        modalImgClass.style.display = "block";
        modalTextClass.style.display = "none";
        document.body.style.overflow = "hidden";
    }
});

dynamicCloseBtn.onclick = function() {
    modalElement.style.display = "none";
    document.body.style.overflow = "auto"; 
}

modalElement.onclick = function(event) {
    if (event.target == modalElement) {
        modalElement.style.display = "none";
        document.body.style.overflow = "auto"; 
    }
}