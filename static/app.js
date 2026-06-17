// =========================================================================
// 1. STATE VARIABLES
// =========================================================================
let currentMode = "movie"; // "movie" or "vibe"
let selectedMovieId = null;
let selectedMovieTitle = "";

// =========================================================================
// 2. DOCUMENT INITIALIZATION
// =========================================================================
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    setupModeToggle();
    setupAutocompleteSearch();
    setupVibeSearch();
}

// =========================================================================
// 3. SEARCH MODE TOGGLING
// =========================================================================
function setupModeToggle() {
    const btnMovie = document.getElementById("mode-movie");
    const btnVibe = document.getElementById("mode-vibe");
    const wrapperMovie = document.getElementById("movie-input-wrapper");
    const wrapperVibe = document.getElementById("vibe-input-wrapper");

    btnMovie.addEventListener("click", () => {
        if (currentMode === "movie") return;
        currentMode = "movie";
        
        btnMovie.classList.add("active");
        btnVibe.classList.remove("active");
        wrapperMovie.classList.add("active");
        wrapperVibe.classList.remove("active");
        
        resetSearchBox();
        clearResults();
        document.getElementById("vibe-query-input").value = "";
    });

    btnVibe.addEventListener("click", () => {
        if (currentMode === "vibe") return;
        currentMode = "vibe";
        
        btnVibe.classList.add("active");
        btnMovie.classList.remove("active");
        wrapperVibe.classList.add("active");
        wrapperMovie.classList.remove("active");
        
        resetSearchBox();
        clearResults();
        document.getElementById("vibe-query-input").value = "";
    });
}

function resetSearchBox() {
    selectedMovieId = null;
    selectedMovieTitle = "";
    const input = document.getElementById("movie-search-input");
    const badge = document.getElementById("selected-movie-badge");
    input.value = "";
    input.style.display = "block";
    badge.style.display = "none";
}

function clearResults() {
    document.getElementById("results-list").innerHTML = "";
    document.getElementById("no-results-placeholder").style.display = "flex";
    document.getElementById("results-header").style.display = "none";
}

// =========================================================================
// 4. AUTOCOMPLETE SEARCH LOGIC
// =========================================================================
function setupAutocompleteSearch() {
    const input = document.getElementById("movie-search-input");
    const box = document.getElementById("search-autocomplete-box");
    const badge = document.getElementById("selected-movie-badge");
    const clearBtn = document.getElementById("clear-selected-movie");
    
    let debounceTimer;
    
    input.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        const query = input.value.trim();
        
        if (query.length < 2) {
            box.style.display = "none";
            return;
        }
        
        debounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(`/api/movies?q=${encodeURIComponent(query)}`);
                const movies = await response.json();
                renderAutocompleteList(movies);
            } catch (err) {
                console.error("Autocomplete search failed:", err);
            }
        }, 250);
    });
    
    // Hide autocomplete list on click outside
    document.addEventListener("click", (e) => {
        if (e.target !== input && e.target !== box) {
            box.style.display = "none";
        }
    });

    clearBtn.addEventListener("click", () => {
        selectedMovieId = null;
        selectedMovieTitle = "";
        badge.style.display = "none";
        input.value = "";
        input.style.display = "block";
        clearResults();
    });
}

function renderAutocompleteList(movies) {
    const box = document.getElementById("search-autocomplete-box");
    box.innerHTML = "";
    
    if (movies.length === 0) {
        box.style.display = "none";
        return;
    }
    
    movies.forEach(movie => {
        const item = document.createElement("div");
        item.className = "autocomplete-item";
        
        item.innerHTML = `
            <div class="item-title">${escapeHTML(movie.title)}</div>
            <div class="item-genres">${escapeHTML(movie.genres.split('|').join(' • '))}</div>
        `;
        
        item.addEventListener("click", () => {
            selectMovie(movie.movieId, movie.title);
            box.style.display = "none";
        });
        
        box.appendChild(item);
    });
    
    box.style.display = "block";
}

function selectMovie(movieId, title) {
    selectedMovieId = movieId;
    selectedMovieTitle = title;
    
    const input = document.getElementById("movie-search-input");
    const badge = document.getElementById("selected-movie-badge");
    const badgeText = document.getElementById("selected-movie-name");
    
    input.style.display = "none";
    badge.style.display = "inline-flex";
    badgeText.innerText = title;
    
    fetchRecommendations({ movieId: movieId });
}

// =========================================================================
// 5. VIBE SEARCH LOGIC
// =========================================================================
function setupVibeSearch() {
    const btn = document.getElementById("vibe-search-btn");
    const input = document.getElementById("vibe-query-input");
    
    btn.addEventListener("click", () => {
        const query = input.value.trim();
        if (query) {
            fetchRecommendations({ query: query });
        }
    });
    
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const query = input.value.trim();
            if (query) {
                fetchRecommendations({ query: query });
            }
        }
    });
}

// =========================================================================
// 6. CALL RECOMMENDATIONS API & RENDER CARDS
// =========================================================================
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function fetchRecommendations(params) {
    showLoader(true);
    const startTime = Date.now();
    
    let url = "";
    if (params.movieId) {
        url = `/api/recommend/semantic?movieId=${params.movieId}&top_k=12`;
    } else if (params.query) {
        url = `/api/recommend/semantic?query=${encodeURIComponent(params.query)}&top_k=12`;
    } else {
        showLoader(false);
        return;
    }
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("API request failed");
        
        const recommendations = await response.json();
        
        // Perceived performance: maintain loader for at least 600ms to allow smooth fade transitions
        const elapsed = Date.now() - startTime;
        if (elapsed < 600) {
            await sleep(600 - elapsed);
        }
        
        renderMovieCards(recommendations);
    } catch (err) {
        console.error("Failed to retrieve recommendations:", err);
        showError("Failed to connect to the recommendation engine. Check if the server is fully ready.");
    } finally {
        showLoader(false);
    }
}

function showLoader(visible) {
    const loader = document.getElementById("results-loader");
    const list = document.getElementById("results-list");
    const placeholder = document.getElementById("no-results-placeholder");
    const header = document.getElementById("results-header");
    
    if (visible) {
        loader.style.display = "flex";
        list.style.display = "none";
        placeholder.style.display = "none";
        header.style.display = "none";
    } else {
        loader.style.display = "none";
        list.style.display = "flex";
    }
}

function showError(msg) {
    const list = document.getElementById("results-list");
    const placeholder = document.getElementById("no-results-placeholder");
    const header = document.getElementById("results-header");
    
    list.innerHTML = "";
    header.style.display = "none";
    placeholder.style.display = "flex";
    document.getElementById("placeholder-text").innerHTML = `<span style="color:var(--accent-primary);"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHTML(msg)}</span>`;
}

function renderMovieCards(movies) {
    const list = document.getElementById("results-list");
    const placeholder = document.getElementById("no-results-placeholder");
    const header = document.getElementById("results-header");
    const countBadge = document.getElementById("results-count");
    
    list.innerHTML = "";
    
    if (!movies || movies.length === 0) {
        placeholder.style.display = "flex";
        document.getElementById("placeholder-text").innerText = "No matching movies found for this input.";
        header.style.display = "none";
        return;
    }
    
    placeholder.style.display = "none";
    header.style.display = "flex";
    countBadge.innerText = `${movies.length} movies`;
    
    movies.forEach((movie, index) => {
        const bar = document.createElement("div");
        bar.className = "movie-bar";
        bar.style.animationDelay = `${index * 0.05}s`;
        
        // Show cosine similarity matching percentage
        const matchPercentage = Math.round(movie.score * 100);
        
        // Format genres properly by splitting on pipe and joining with bullets
        const formattedGenres = movie.genres.split('|').join(' • ');
        
        bar.innerHTML = `
            <div class="bar-info">
                <h4 class="bar-title">${escapeHTML(movie.title)}</h4>
                <div class="bar-genres">${escapeHTML(formattedGenres)}</div>
            </div>
            <div class="bar-stats">
                <span class="bar-rating">
                    <i class="fa-solid fa-star"></i>
                    <span>${movie.rating ? movie.rating.toFixed(1) : "N/A"}</span>
                </span>
                <span class="bar-score">${matchPercentage}% Match</span>
            </div>
            <div class="bar-links">
                <a href="${movie.imdbUrl}" target="_blank" class="bar-btn imdb" title="View on IMDb"><i class="fa-brands fa-imdb"></i></a>
                <a href="${movie.tmdbUrl}" target="_blank" class="bar-btn tmdb" title="View on TMDb">TMDb</a>
            </div>
        `;
        
        list.appendChild(bar);
    });
}

/**
 * Creates a unique linear HSL gradient string by hashing the movie title.
 * Makes cards look colorful and professional without needing poster images.
 */
function generateGradient(title) {
    let hash = 0;
    for (let i = 0; i < title.length; i++) {
        hash = title.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const h1 = Math.abs(hash % 360);
    const h2 = (h1 + 55) % 360; // Complementary offset
    
    return `linear-gradient(135deg, hsl(${h1}, 70%, 22%) 0%, hsl(${h2}, 55%, 11%) 100%)`;
}

// =========================================================================
// 7. UTILITY FUNCTIONS
// =========================================================================
function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
