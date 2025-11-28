
    (function() {
      const outerHeaders = document.querySelectorAll('.tab-headers li');
      const outerPanels = document.querySelectorAll('.tabs > .tab-panels > .tab-panel');

      // Inner tab switches
      document.querySelectorAll('.subtabs').forEach(block => {
        const sheaders = block.querySelectorAll('.subtab-headers .subtab');
        const spans = block.querySelectorAll('.tab-panels > .tab-panel');
        function sactivate(targetId) {
          sheaders.forEach(h => h.classList.toggle('active', h.dataset.target === targetId));
          spans.forEach(p => {
            const isMatch = p.id === targetId;
            p.classList.toggle('active', isMatch);
            p.style.display = isMatch ? 'block' : 'none';
          });
        }
        sheaders.forEach(h => h.addEventListener('click', () => sactivate(h.dataset.target)));
        if (sheaders.length) sactivate(sheaders[0].dataset.target);
      });

      function activateOuter(idx) {
        outerHeaders.forEach((h,i)=>h.classList.toggle('active', i===idx));
        outerPanels.forEach((p,i)=> {
          const isActive = i===idx;
          p.classList.toggle('active', isActive);
          p.style.display = isActive ? 'block' : 'none';
          if (isActive) {
            const firstSub = p.querySelector('.subtab-headers .subtab');
            if (firstSub) firstSub.click();
          }
        });
      }

      outerHeaders.forEach((h,i)=>h.addEventListener('click', ()=>activateOuter(i)));
      if (outerHeaders.length) {
        activateOuter(0);
      } else {
        outerPanels.forEach(p => {
          p.classList.add('active');
          p.style.display = 'block';
        });
      }

      // Color control por tarjeta
      document.querySelectorAll('.plot-card').forEach(card => {
        const sid = card.dataset.sid || '';
        const controls = card.querySelector('.color-controls');
        const panels = card.querySelectorAll('.subtab-panels .subtab-panel');
        const modeSel = card.querySelector(`#${sid}-mode`);
        const expSlider = card.querySelector(`#${sid}-exp`);
        const expVal = card.querySelector(`#${sid}-exp-val`);
        const defaultMode = controls ? controls.dataset.defaultMode || 'pair_exp' : 'pair_exp';
        const defaultExp = controls ? parseFloat(controls.dataset.defaultExp || '1') : 1;
        if (modeSel && !modeSel.value) modeSel.value = defaultMode;
        if (expSlider && !expSlider.value) expSlider.value = defaultExp.toFixed(2);

        function showPanel() {
          const mode = modeSel ? modeSel.value : defaultMode;
          let optionDefaultExp = defaultExp;
          if (modeSel) {
            const opt = modeSel.options[modeSel.selectedIndex];
            if (opt && opt.dataset.defaultExp) {
              const candidate = parseFloat(opt.dataset.defaultExp);
              if (Number.isFinite(candidate)) optionDefaultExp = candidate;
            }
          }
          const needsExp = (mode === 'pair_exp' || mode === 'types_exp');
          let exp = needsExp ? (expSlider ? parseFloat(expSlider.value) : optionDefaultExp) : optionDefaultExp;
          if (!Number.isFinite(exp)) exp = optionDefaultExp;
          const code = String(Math.round(exp * 100)).padStart(3, '0');
          const target = needsExp ? `${sid}-${mode}_${code}` : `${sid}-${mode}`;
          panels.forEach(p => {
            const isTarget = p.id === target;
            p.classList.toggle('active', isTarget);
            p.style.display = isTarget ? 'block' : 'none';
          });
          if (expSlider) {
            if (needsExp) {
              expSlider.removeAttribute('disabled');
              expSlider.value = exp.toFixed(2);
              if (expVal) expVal.textContent = exp.toFixed(2);
            } else {
              expSlider.setAttribute('disabled', 'disabled');
              expSlider.value = optionDefaultExp.toFixed(2);
              if (expVal) expVal.textContent = '--';
            }
          } else if (expVal) {
            expVal.textContent = needsExp ? exp.toFixed(2) : '--';
          }
        }

        if (modeSel) modeSel.addEventListener('change', () => showPanel());
        if (expSlider) {
          expSlider.addEventListener('input', () => showPanel());
          expSlider.addEventListener('change', () => showPanel());
        }
        showPanel();
      });

      function getBaseMarkerOpacities(gd, traceIndex, trace) {
        if (!gd.__baseMarkerOpacities) gd.__baseMarkerOpacities = {};
        const cache = gd.__baseMarkerOpacities;
        const length = Array.isArray(trace.customdata) ? trace.customdata.length : (Array.isArray(trace.x) ? trace.x.length : 0);
        if (!length) return null;
        if (cache[traceIndex] && cache[traceIndex].length === length) {
          return cache[traceIndex];
        }
        let baseArray;
        const marker = trace.marker || {};
        if (Array.isArray(marker.opacity) && marker.opacity.length === length) {
          baseArray = marker.opacity.slice();
        } else {
          const base = typeof marker.opacity === 'number' ? marker.opacity : 0.6;
          baseArray = new Array(length).fill(base);
        }
        cache[traceIndex] = baseArray;
        return baseArray;
      }

      function applyGlobalIdHighlight(gd, activeIds, fadeFactor = 0.1) {
        const ids = activeIds && activeIds.size ? activeIds : null;
        const indices = [];
        const payload = [];
        gd.data.forEach((trace, idx) => {
          if (!Array.isArray(trace.customdata) || !trace.customdata.length) return;
          const baseOpacities = getBaseMarkerOpacities(gd, idx, trace);
          if (!baseOpacities) return;
          indices.push(idx);
          payload.push(trace.customdata.map((row, pointIdx) => {
            const gid = row && row.length >= 8 ? Number(row[7]) : NaN;
            if (ids && (!Number.isFinite(gid) || !ids.has(gid))) {
              return baseOpacities[pointIdx] * fadeFactor;
            }
            return baseOpacities[pointIdx];
          }));
        });
        if (indices.length) {
          Plotly.restyle(gd, {'marker.opacity': payload}, indices);
        }
      }

      function findPointCoordinates(gd, targetId) {
        if (!Number.isFinite(targetId)) return null;
        for (let i = 0; i < gd.data.length; i++) {
          const trace = gd.data[i];
          const cd = trace.customdata || [];
          const xs = trace.x || [];
          const ys = trace.y || [];
          for (let j = 0; j < cd.length; j++) {
            const row = cd[j];
            if (row && row.length >= 8 && Number(row[7]) === Number(targetId)) {
              return { x: xs[j], y: ys[j] };
            }
          }
        }
        return null;
      }

      function setupFamilyHighlight(gd) {
        if (!gd || gd.__familyHighlightBound) return;
        const info = gd.layout && gd.layout.meta && gd.layout.meta.familyHighlight;
        if (!info || !info.enabled) return;
        gd.__familyHighlightBound = true;
        let activeTag = null;

        function applySelection(tag) {
          if (tag === activeTag) return;
          activeTag = tag;
          gd.data.forEach((trace, traceIndex) => {
            const custom = trace.customdata || [];
            if (!custom.length) {
              Plotly.restyle(gd, {selectedpoints: [null]}, [traceIndex]);
              return;
            }
            const matches = [];
            if (tag) {
              const tagStr = String(tag);
              for (let i = 0; i < custom.length; i++) {
                const row = custom[i];
                if (!row) continue;
                if (String(row[0]) === tagStr) {
                  matches.push(i);
                }
              }
            }
            Plotly.restyle(gd, {selectedpoints: [matches.length ? matches : null]}, [traceIndex]);
          });
        }

        gd.on('plotly_hover', ev => {
          const pt = ev.points && ev.points[0];
          if (!pt || !pt.customdata) {
            applySelection(null);
            return;
          }
          const familySize = parseInt(pt.customdata[2], 10) || 0;
          if (familySize < 2) {
            applySelection(null);
            return;
          }
          applySelection(String(pt.customdata[0]));
        });

        gd.on('plotly_unhover', () => applySelection(null));
        gd.on('plotly_click', () => applySelection(null));
        applySelection(null);
      }

      function getFilterDataset(gd) {
        if (!gd || !gd.layout || !gd.layout.meta) return null;
        const meta = gd.layout.meta;
        return meta.filterDataset || null;
      }

      function ensureSourceCache(source) {
        if (!source) {
          return {cardinality: [], intervalPattern: [], pitchClass: [], maxInternalInterval: []};
        }
        if (!source._filterCache) {
          const filterValues = source.filterValues || {};
          const normaliseList = values => {
            if (!Array.isArray(values)) return [];
            return values.map(value => (value === null || value === undefined ? null : String(value)));
          };
          const cardinality = normaliseList(filterValues.cardinality);
          const intervalPattern = normaliseList(filterValues.intervalPattern);
          const pitchClass = Array.isArray(filterValues.pitchClass)
            ? filterValues.pitchClass.map(entry => {
                if (Array.isArray(entry)) {
                  return entry
                    .filter(value => value !== null && value !== undefined)
                    .map(value => String(value));
                }
                if (entry === null || entry === undefined) {
                  return [];
                }
                return [String(entry)];
              })
            : [];
          const maxInternalInterval = Array.isArray(filterValues.maxInternalInterval)
            ? filterValues.maxInternalInterval.map(value => {
                const num = Number(value);
                return Number.isFinite(num) ? num : null;
              })
            : [];
          source._filterCache = {
            cardinality,
            intervalPattern,
            pitchClass,
            maxInternalInterval,
          };
        }
        return source._filterCache;
      }

      function applyFiltersToFigure(gd, filters) {
        const dataset = getFilterDataset(gd);
        if (!dataset || !Array.isArray(dataset.traceSources)) return;
        const cardinalFilter = filters.cardinality && filters.cardinality.size ? filters.cardinality : null;
        const intervalFilter = filters.intervalPattern && filters.intervalPattern.size ? filters.intervalPattern : null;
        const pitchFilter = filters.pitchClass && filters.pitchClass.size ? filters.pitchClass : null;
        const internalFilter = Number.isFinite(filters.maxInternalInterval)
          ? Number(filters.maxInternalInterval)
          : null;
        const noFilters = !cardinalFilter && !intervalFilter && !pitchFilter && internalFilter === null;

        const updates = {
          x: [],
          y: [],
          text: [],
          customdata: [],
          marker: [],
        };
        const indices = [];

        dataset.traceSources.forEach(source => {
          const traceIndex = typeof source.traceIndex === 'number' ? source.traceIndex : parseInt(source.traceIndex, 10);
          if (!Number.isFinite(traceIndex)) return;
          const xSource = Array.isArray(source.x) ? source.x : [];
          const ySource = Array.isArray(source.y) ? source.y : [];
          const textSource = Array.isArray(source.text) ? source.text : [];
          const customSource = Array.isArray(source.customdata) ? source.customdata : [];
          const colorSource = Array.isArray(source.colors) ? source.colors : [];

          if (noFilters) {
            const baseMarker = Object.assign({}, source.baseMarker || {});
            baseMarker.color = colorSource.slice();
            updates.x.push(xSource.slice());
            updates.y.push(ySource.slice());
            updates.text.push(textSource.slice());
            updates.customdata.push(customSource.slice());
            updates.marker.push(baseMarker);
            indices.push(traceIndex);
            return;
          }

          const cache = ensureSourceCache(source);
          const cardinalVals = cache.cardinality;
          const intervalVals = cache.intervalPattern;
          const pitchVals = cache.pitchClass;
          const internalVals = cache.maxInternalInterval;

          const selected = [];
          for (let i = 0; i < xSource.length; i++) {
            const cardVal = cardinalVals[i];
            const intervalVal = intervalVals[i];
            if (cardinalFilter && (!cardVal || !cardinalFilter.has(cardVal))) continue;
            if (intervalFilter && (!intervalVal || !intervalFilter.has(intervalVal))) continue;
            if (pitchFilter) {
              const rawPitch = Array.isArray(pitchVals[i]) ? pitchVals[i] : [];
              let intersects = false;
              for (let j = 0; j < rawPitch.length; j++) {
                const pitchVal = rawPitch[j];
                if (pitchFilter.has(pitchVal)) {
                  intersects = true;
                  break;
                }
              }
              if (!intersects) continue;
            }
            if (internalFilter !== null) {
              const rawInternal = internalVals[i];
              const numericInternal = Number(rawInternal);
              if (!Number.isFinite(numericInternal) || numericInternal > internalFilter) continue;
            }
            selected.push(i);
          }

          const pick = arr => selected.map(idx => arr[idx]);
          const baseMarker = Object.assign({}, source.baseMarker || {});
          baseMarker.color = pick(colorSource);
          updates.x.push(pick(xSource));
          updates.y.push(pick(ySource));
          updates.text.push(pick(textSource));
          updates.customdata.push(pick(customSource));
          updates.marker.push(baseMarker);
          indices.push(traceIndex);
        });

        if (indices.length) {
          Plotly.restyle(gd, updates, indices);
        }
      }

      function registerCardFilters(card) {
        const figures = Array.from(card.querySelectorAll('.js-plotly-plot'));
        if (!figures.length) return;
        const detailPanel = card.querySelector('.detail-panel');
        const container = document.createElement('div');
        container.className = 'filter-controls';
        const title = document.createElement('span');
        title.className = 'filter-title';
        title.textContent = 'Filtros dinámicos';
        container.appendChild(title);
        if (detailPanel) {
          card.insertBefore(container, detailPanel);
        } else {
          card.appendChild(container);
        }

        const state = {
          cardinality: new Set(),
          intervalPattern: new Set(),
          pitchClass: new Set(),
          maxInternalInterval: null,
        };
        const metadata = {
          cardinality: new Set(),
          intervalPattern: new Set(),
          pitchClass: new Set(),
          maxInternalInterval: null,
        };
        const controls = {
          cardinality: new Map(),
          intervalPattern: new Map(),
          pitchClass: new Map(),
          maxInternalInterval: null,
        };

        function ensureFilterControl(def, field) {
          if (!def) return;
          const fieldset = document.createElement('fieldset');
          fieldset.className = 'filter-group';
          const legend = document.createElement('legend');
          legend.textContent = def.label || field;
          fieldset.appendChild(legend);

          if (def.type === 'numeric') {
            const min = Number.isFinite(def.min) ? Number(def.min) : 0;
            const max = Number.isFinite(def.max) ? Number(def.max) : min;
            const step = Number.isFinite(def.step) && def.step > 0 ? Number(def.step) : 1;
            const defaultValue = Number.isFinite(def.default) ? Number(def.default) : max;

            metadata[field] = { min, max, step, defaultValue };

            const wrapper = document.createElement('div');
            wrapper.className = 'filter-numeric';

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = String(min);
            slider.max = String(max);
            slider.step = String(step);
            slider.value = String(defaultValue);

            const numberInput = document.createElement('input');
            numberInput.type = 'number';
            numberInput.min = String(min);
            numberInput.max = String(max);
            numberInput.step = String(step);
            numberInput.value = String(defaultValue);

            const clearButton = document.createElement('button');
            clearButton.type = 'button';
            clearButton.textContent = 'Limpiar';
            clearButton.className = 'filter-clear';

            const applyValue = value => {
              if (value === null) {
                state[field] = null;
                numberInput.value = '';
                slider.value = String(defaultValue);
              } else {
                const bounded = Math.max(min, Math.min(max, value));
                slider.value = String(bounded);
                numberInput.value = String(bounded);
                state[field] = bounded;
              }
              applyAll();
            };

            slider.addEventListener('input', () => {
              const val = Number(slider.value);
              if (Number.isFinite(val)) {
                numberInput.value = slider.value;
                state[field] = val;
                applyAll();
              }
            });

            numberInput.addEventListener('change', () => {
              const raw = numberInput.value.trim();
              if (!raw) {
                applyValue(null);
                return;
              }
              const parsed = Number(raw);
              if (Number.isFinite(parsed)) {
                applyValue(parsed);
              } else {
                numberInput.value = state[field] === null ? '' : String(state[field]);
              }
            });

            clearButton.addEventListener('click', () => applyValue(null));

            wrapper.appendChild(slider);
            wrapper.appendChild(numberInput);
            wrapper.appendChild(clearButton);
            fieldset.appendChild(wrapper);

            controls[field] = { slider, numberInput, clearButton };
            state[field] = defaultValue;
            container.appendChild(fieldset);
            return;
          }

          if (!Array.isArray(def.options) || !def.options.length) return;

          const fieldMetadata = metadata[field] || new Set();
          metadata[field] = fieldMetadata;
          const fieldControls = controls[field] || new Map();
          controls[field] = fieldControls;

          let optionElements = null;
          let subgroups = null;

          const registerOption = (parent, opt, subgroup = null) => {
            const idStr = String(opt.id);
            fieldMetadata.add(idStr);

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = idStr;
            checkbox.checked = opt.default !== false;

            if (checkbox.checked) {
              state[field].add(idStr);
            }

            fieldControls.set(idStr, checkbox);

            const label = document.createElement('label');
            label.className = 'filter-option';
            const labelText = opt.label || opt.id;
            const rawCount = typeof opt.count === 'number' ? opt.count : NaN;
            const count = Number.isFinite(rawCount) && rawCount > 0 ? rawCount : null;
            const searchText = count ? `${labelText} ${count}` : String(labelText);
            label.dataset.optionText = searchText.toLowerCase();

            label.appendChild(checkbox);
            const textSpan = document.createElement('span');
            textSpan.textContent = labelText;
            label.appendChild(textSpan);

            if (count) {
              const countSpan = document.createElement('span');
              countSpan.className = 'option-count';
              countSpan.textContent = `(${count})`;
              label.appendChild(countSpan);
            }

            checkbox.addEventListener('change', () => {
              if (checkbox.checked) {
                state[field].add(idStr);
              } else {
                state[field].delete(idStr);
              }
              applyAll();
            });

            parent.appendChild(label);
            if (optionElements && subgroup) {
              optionElements.push({label, subgroup});
            }
          };

          if (field === 'intervalPattern') {
            optionElements = [];
            subgroups = [];

            const setAll = checked => {
              const fieldState = state[field];
              if (checked) {
                fieldState.clear();
                fieldControls.forEach((checkbox, id) => {
                  checkbox.checked = true;
                  fieldState.add(id);
                });
              } else {
                fieldControls.forEach(checkbox => {
                  checkbox.checked = false;
                });
                fieldState.clear();
              }
              applyAll();
            };

            const actions = document.createElement('div');
            actions.className = 'filter-actions';
            const selectAllBtn = document.createElement('button');
            selectAllBtn.type = 'button';
            selectAllBtn.textContent = 'Seleccionar todo';
            selectAllBtn.addEventListener('click', () => setAll(true));
            const clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.textContent = 'Limpiar';
            clearBtn.addEventListener('click', () => setAll(false));
            actions.appendChild(selectAllBtn);
            actions.appendChild(clearBtn);
            fieldset.appendChild(actions);

            const searchWrapper = document.createElement('div');
            searchWrapper.className = 'filter-search';
            const searchInput = document.createElement('input');
            searchInput.type = 'search';
            searchInput.placeholder = 'Buscar patrón...';
            searchWrapper.appendChild(searchInput);
            fieldset.appendChild(searchWrapper);

            const groupsContainer = document.createElement('div');
            groupsContainer.className = 'filter-subgroup-container';
            fieldset.appendChild(groupsContainer);

            const groups = new Map();
            def.options.forEach(opt => {
              const cardValue = typeof opt.cardinality === 'number' ? opt.cardinality : parseInt(opt.cardinality, 10);
              const key = Number.isFinite(cardValue) ? String(cardValue) : 'Otros';
              if (!groups.has(key)) {
                groups.set(key, []);
              }
              groups.get(key).push(opt);
            });

            const sortedKeys = Array.from(groups.keys()).sort((a, b) => {
              const aNum = parseInt(a, 10);
              const bNum = parseInt(b, 10);
              const aValid = !Number.isNaN(aNum);
              const bValid = !Number.isNaN(bNum);
              if (aValid && bValid) return aNum - bNum;
              if (aValid) return -1;
              if (bValid) return 1;
              return String(a).localeCompare(String(b), 'es');
            });

            sortedKeys.forEach(key => {
              const opts = groups.get(key) || [];
              const subgroup = document.createElement('details');
              subgroup.className = 'filter-subgroup';
              const keyNum = parseInt(key, 10);
              const isNumeric = !Number.isNaN(keyNum);
              const defaultOpen = false;
              subgroup.open = defaultOpen;
              subgroup.dataset.defaultOpen = defaultOpen ? '1' : '0';

              const summary = document.createElement('summary');
              const totalCount = opts.reduce((acc, option) => {
                const raw = typeof option.count === 'number' ? option.count : NaN;
                return acc + (Number.isFinite(raw) ? raw : 0);
              }, 0);
              const parts = [];
              if (isNumeric) {
                parts.push(`${keyNum} ${keyNum === 1 ? 'nota' : 'notas'}`);
              } else {
                parts.push(key);
              }
              parts.push(`${opts.length} ${opts.length === 1 ? 'patrón' : 'patrones'}`);
              if (totalCount > 0) {
                parts.push(`${totalCount} ${totalCount === 1 ? 'acorde' : 'acordes'}`);
              }
              summary.textContent = parts.join(' · ');
              subgroup.appendChild(summary);

              const wrapper = document.createElement('div');
              wrapper.className = 'filter-options-grid';
              opts.forEach(option => registerOption(wrapper, option, subgroup));
              subgroup.appendChild(wrapper);
              groupsContainer.appendChild(subgroup);
              subgroups.push(subgroup);
            });

            const handleSearch = () => {
              const term = searchInput.value.trim().toLowerCase();
              const visibility = new Map();
              subgroups.forEach(subgroup => visibility.set(subgroup, false));
              optionElements.forEach(item => {
                const text = item.label.dataset.optionText || '';
                const matches = !term || text.includes(term);
                item.label.style.display = matches ? '' : 'none';
                if (matches) {
                  visibility.set(item.subgroup, true);
                }
              });
              subgroups.forEach(subgroup => {
                const visible = visibility.get(subgroup);
                subgroup.style.display = visible ? '' : 'none';
                subgroup.open = term ? !!visible : subgroup.dataset.defaultOpen === '1';
              });
            };
            searchInput.addEventListener('input', handleSearch);
          } else if (field === 'pitchClass') {
            const details = document.createElement('details');
            details.className = 'filter-subgroup';
            details.open = false;
            details.dataset.defaultOpen = '0';
            const summary = document.createElement('summary');
            const totalOptions = Array.isArray(def.options) ? def.options.length : 0;
            summary.textContent = `Opciones (${totalOptions})`;
            details.appendChild(summary);
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'filter-options-vertical';
            def.options.forEach(opt => registerOption(optionsContainer, opt, details));
            details.appendChild(optionsContainer);
            fieldset.appendChild(details);
          } else {
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'filter-options-vertical';
            def.options.forEach(opt => registerOption(optionsContainer, opt));
            fieldset.appendChild(optionsContainer);
          }

          container.appendChild(fieldset);
        }

        const getEffectiveFilter = field => {
          if (field === 'maxInternalInterval') {
            const value = state[field];
            if (!Number.isFinite(value)) return null;
            const info = metadata[field];
            if (info && Number.isFinite(info.defaultValue) && value >= info.defaultValue) {
              return null;
            }
            return value;
          }
          const selected = state[field];
          if (!selected || !selected.size) return null;
          const available = metadata[field];
          const availableSize = available ? available.size : 0;
          if (availableSize && selected.size >= availableSize) {
            return null;
          }
          return selected;
        };

        const buildSignature = filters => {
          const parts = [];
          ['cardinality', 'intervalPattern', 'pitchClass'].forEach(field => {
            const set = filters[field];
            if (!set || !set.size) {
              parts.push(`${field}:*`);
            } else {
              const values = Array.from(set).sort();
              parts.push(`${field}:${values.join(',')}`);
            }
          });
          const numeric = filters.maxInternalInterval;
          parts.push(`maxInternalInterval:${numeric === null || numeric === undefined ? '*' : numeric}`);
          return parts.join('|');
        };

        const applyAll = () => {
          const filters = {
            cardinality: getEffectiveFilter('cardinality'),
            intervalPattern: getEffectiveFilter('intervalPattern'),
            pitchClass: getEffectiveFilter('pitchClass'),
            maxInternalInterval: getEffectiveFilter('maxInternalInterval'),
          };
          const signature = buildSignature(filters);
          figures.forEach(gd => {
            if (gd.__filterSignature === signature) return;
            applyFiltersToFigure(gd, filters);
            gd.__filterSignature = signature;
          });
        };

        const attachUI = dataset => {
          const fields = dataset && dataset.fields ? dataset.fields : {};
          ensureFilterControl(fields.cardinality, 'cardinality');
          ensureFilterControl(fields.maxInternalInterval, 'maxInternalInterval');
          ensureFilterControl(fields.intervalPattern, 'intervalPattern');
          ensureFilterControl(fields.pitchClass, 'pitchClass');
        };

        let pending = figures.length;
        figures.forEach(gd => {
          const dataset = getFilterDataset(gd);
          if (dataset) {
            pending -= 1;
            if (pending === figures.length - 1) {
              attachUI(dataset);
            }
            if (pending === 0) {
              applyAll();
            }
          } else {
            const handler = () => {
              const ds = getFilterDataset(gd);
              if (!ds) return;
              gd.removeListener('plotly_afterplot', handler);
              pending -= 1;
              if (pending === figures.length - 1) {
                attachUI(ds);
              }
              if (pending === 0) {
                applyAll();
              }
            };
            gd.on('plotly_afterplot', handler);
          }
        });
      }

      function setupInversionHighlight(gd) {
        if (!gd || gd.__inversionHighlightBound) return;
        gd.__inversionHighlightBound = true;

        const card = gd.closest('.plot-card');
        const musicalToggle = card.querySelector('.inversion-toggle[data-inversion-type="musical"]');
        const structuralToggle = card.querySelector('.inversion-toggle[data-inversion-type="structural"]');

        let activeInversions = new Set();
        let lastMusical = new Set();
        let lastStructural = new Set();
        let hasHover = false;
        let currentHover = { id: null, x: null, y: null };

        function togglesEnabled() {
            return (musicalToggle && musicalToggle.checked) || (structuralToggle && structuralToggle.checked);
        }

        function ensureOverlay() {
            if (gd.__invOverlay && gd.__invOverlay.ready) {
                return Promise.resolve(gd.__invOverlay);
            }
            if (gd.__invOverlayPending) {
                return gd.__invOverlayPending;
            }
            const baseIndex = gd.data.length;
            const overlayTraces = [
                {
                    type: 'scatter', mode: 'markers', name: '', showlegend: false, hoverinfo: 'skip',
                    x: [], y: [],
                    marker: {
                        size: 18,
                        color: 'rgba(0, 184, 217, 0.28)',
                        line: { color: '#002C3A', width: 2.5 },
                        symbol: 'circle'
                    }
                },
                {
                    type: 'scatter', mode: 'lines', name: '', showlegend: false, hoverinfo: 'skip',
                    x: [], y: [],
                    line: { color: '#00B8D9', width: 2.4, dash: 'solid' }
                },
                {
                    type: 'scatter', mode: 'lines', name: '', showlegend: false, hoverinfo: 'skip',
                    x: [], y: [],
                    line: { color: '#FF2D6D', width: 2.4, dash: 'dash' }
                }
            ];
            gd.__invOverlayPending = Plotly.addTraces(gd, overlayTraces).then(() => {
                gd.__invOverlay = {
                    markersIdx: baseIndex,
                    linksMusIdx: baseIndex + 1,
                    linksStrIdx: baseIndex + 2,
                    ready: true,
                };
                gd.__invOverlayPending = null;
                return gd.__invOverlay;
            }).catch(() => {
                gd.__invOverlayPending = null;
                gd.__invOverlay = null;
                return null;
            });
            return gd.__invOverlayPending;
        }

        function hideOverlay() {
            const overlays = gd.__invOverlay;
            if (!overlays || typeof overlays.markersIdx !== 'number') return;
            Plotly.restyle(gd, { x: [[]], y: [[]], visible: [false] }, [overlays.markersIdx]);
            Plotly.restyle(gd, { x: [[]], y: [[]], visible: [false] }, [overlays.linksMusIdx]);
            Plotly.restyle(gd, { x: [[]], y: [[]], visible: [false] }, [overlays.linksStrIdx]);
        }

        function idToXY(targetId) {
            return findPointCoordinates(gd, targetId);
        }

        function updateOverlay(musicalSet, structuralSet) {
            if (!togglesEnabled() || (!musicalSet.size && !structuralSet.size) || !Number.isFinite(currentHover.id)) {
                hideOverlay();
                return;
            }
            ensureOverlay().then(overlays => {
                if (!overlays) return;
                const union = new Set();
                musicalSet.forEach(id => union.add(id));
                structuralSet.forEach(id => union.add(id));
                union.add(currentHover.id);

                const mx = [], my = [];
                union.forEach(id => {
                    const pos = idToXY(id);
                    if (pos) { mx.push(pos.x); my.push(pos.y); }
                });

                const hx = currentHover.x;
                const hy = currentHover.y;
                const linksSolidX = [];
                const linksSolidY = [];
                const linksDashX = [];
                const linksDashY = [];
                if (hx !== null && hy !== null) {
                    musicalSet.forEach(id => {
                        const pos = idToXY(id);
                        if (pos) { linksSolidX.push(hx, pos.x, null); linksSolidY.push(hy, pos.y, null); }
                    });
                    structuralSet.forEach(id => {
                        const pos = idToXY(id);
                        if (pos) { linksDashX.push(hx, pos.x, null); linksDashY.push(hy, pos.y, null); }
                    });
                }

                Plotly.restyle(gd, { x: [mx], y: [my], visible: [mx.length ? true : false] }, [overlays.markersIdx]);
                Plotly.restyle(gd, { x: [linksSolidX], y: [linksSolidY], visible: [linksSolidX.length ? true : false] }, [overlays.linksMusIdx]);
                Plotly.restyle(gd, { x: [linksDashX], y: [linksDashY], visible: [linksDashX.length ? true : false] }, [overlays.linksStrIdx]);
            });
        }

        function applyInversionHighlight(useMusicalSet, useStructuralSet) {
            const ids = activeInversions.size ? new Set(Array.from(activeInversions).map(Number)) : null;
            applyGlobalIdHighlight(gd, ids, 0.1);
            updateOverlay(useMusicalSet || new Set(), useStructuralSet || new Set());
        }

        function recomputeActive() {
            activeInversions.clear();
            if (!hasHover || !togglesEnabled()) {
                applyInversionHighlight(new Set(), new Set());
                hideOverlay();
                return;
            }
            const useMusical = (musicalToggle && musicalToggle.checked) ? new Set(lastMusical) : new Set();
            const useStructural = (structuralToggle && structuralToggle.checked) ? new Set(lastStructural) : new Set();
            useMusical.forEach(id => activeInversions.add(id));
            useStructural.forEach(id => activeInversions.add(id));
            if (Number.isFinite(currentHover.id)) {
                activeInversions.add(currentHover.id);
            }
            applyInversionHighlight(useMusical, useStructural);
        }

        gd.on('plotly_hover', ev => {
            const pt = ev.points && ev.points[0];
            if (!pt || !pt.customdata) {
                hasHover = false;
                lastMusical = new Set();
                lastStructural = new Set();
                currentHover = { id: null, x: null, y: null };
                recomputeActive();
                return;
            }
            lastMusical = new Set(pt.customdata[5] || []);
            lastStructural = new Set(pt.customdata[6] || []);
            currentHover = {
                id: typeof pt.customdata[7] === 'number' ? pt.customdata[7] : parseInt(pt.customdata[7], 10),
                x: pt.x,
                y: pt.y,
            };
            hasHover = true;
            recomputeActive();
        });

        gd.on('plotly_unhover', () => {
            hasHover = false;
            lastMusical = new Set();
            lastStructural = new Set();
            currentHover = { id: null, x: null, y: null };
            recomputeActive();
        });

        if (musicalToggle) musicalToggle.addEventListener('change', () => recomputeActive());
        if (structuralToggle) structuralToggle.addEventListener('change', () => recomputeActive());
      }

      function resolveDefaultSubstitutionProfile(neighborsByProfile, profileConfig) {
        if (!neighborsByProfile) return null;
        const keys = Object.keys(neighborsByProfile);
        if (!keys.length) return null;
        if (profileConfig && profileConfig.default && neighborsByProfile[profileConfig.default]) {
          return profileConfig.default;
        }
        return keys[0];
      }

      function populateSubstitutionProfileSelect(select, neighborsByProfile, profileConfig, activeKey) {
        if (!select) return;
        select.innerHTML = '';
        if (!neighborsByProfile) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No disponible';
          select.appendChild(opt);
          select.disabled = true;
          return;
        }
        const keys = Object.keys(neighborsByProfile);
        if (!keys.length) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No disponible';
          select.appendChild(opt);
          select.disabled = true;
          return;
        }
        const metaProfiles = (profileConfig && profileConfig.profiles) || {};
        keys.forEach(key => {
          const opt = document.createElement('option');
          opt.value = key;
          const info = metaProfiles[key];
          opt.textContent = (info && info.label) ? info.label : key;
          select.appendChild(opt);
        });
        if (activeKey && neighborsByProfile[activeKey]) {
          select.value = activeKey;
        } else {
          select.selectedIndex = 0;
        }
        select.disabled = select.options.length <= 1;
      }

      function setupSubstitutionHighlight(gd) {
        if (!gd || gd.__substitutionHighlightBound) return;
        const card = gd.closest('.plot-card');
        const toggle = card.querySelector('.substitution-toggle');
        const profileSelect = card.querySelector('.substitution-profile');
        const neighborsByProfile = gd.layout && gd.layout.meta && gd.layout.meta.substitutionNeighbors;
        const profileConfig = gd.layout && gd.layout.meta && gd.layout.meta.substitutionProfiles;
        if (!toggle || !neighborsByProfile) return;
        const availableProfiles = Object.keys(neighborsByProfile);
        if (!availableProfiles.length) return;
        gd.__substitutionHighlightBound = true;

        const defaultProfile = resolveDefaultSubstitutionProfile(neighborsByProfile, profileConfig);
        if (!gd.__substitutionProfile || !neighborsByProfile[gd.__substitutionProfile]) {
          gd.__substitutionProfile = defaultProfile;
        }

        if (profileSelect) {
          populateSubstitutionProfileSelect(profileSelect, neighborsByProfile, profileConfig, gd.__substitutionProfile);
          profileSelect.addEventListener('change', () => {
            const selected = profileSelect.value;
            if (selected && neighborsByProfile[selected]) {
              gd.__substitutionProfile = selected;
            } else {
              gd.__substitutionProfile = resolveDefaultSubstitutionProfile(neighborsByProfile, profileConfig);
              if (gd.__substitutionProfile && neighborsByProfile[gd.__substitutionProfile]) {
                profileSelect.value = gd.__substitutionProfile;
              }
            }
            card.dispatchEvent(new CustomEvent('substitutionProfileChanged', { detail: { profile: gd.__substitutionProfile } }));
            if (toggle.checked && Number.isFinite(lastHoverId)) {
              applyForId(lastHoverId);
            } else if (!toggle.checked) {
              hideOverlay();
              applyGlobalIdHighlight(gd, null);
            }
          });
        }

        let lastHoverId = null;

        function ensureSubOverlay() {
          if (gd.__subOverlay && gd.__subOverlay.ready) {
            return Promise.resolve(gd.__subOverlay);
          }
          if (gd.__subOverlayPending) {
            return gd.__subOverlayPending;
          }
          const overlayTrace = {
            type: 'scatter',
            mode: 'lines',
            name: '',
            showlegend: false,
            hoverinfo: 'skip',
            x: [],
            y: [],
            line: { color: '#007BFF', width: 2, dash: 'dot' },
            visible: false,
          };
          gd.__subOverlayPending = Plotly.addTraces(gd, [overlayTrace]).then(() => {
            gd.__subOverlay = {
              lineIdx: gd.data.length - 1,
              ready: true,
            };
            gd.__subOverlayPending = null;
            return gd.__subOverlay;
          }).catch(() => {
            gd.__subOverlayPending = null;
            gd.__subOverlay = null;
            return null;
          });
          return gd.__subOverlayPending;
        }

        function hideOverlay() {
          const overlay = gd.__subOverlay;
          if (!overlay || typeof overlay.lineIdx !== 'number') return;
          Plotly.restyle(gd, { x: [[]], y: [[]], visible: [false] }, [overlay.lineIdx]);
        }

        function drawLines(sourceId, neighborIds) {
          ensureSubOverlay().then(overlay => {
            if (!overlay || typeof overlay.lineIdx !== 'number') return;
            const origin = findPointCoordinates(gd, sourceId);
            if (!origin) {
              hideOverlay();
              return;
            }
            const xs = [];
            const ys = [];
            neighborIds.forEach(id => {
              const dest = findPointCoordinates(gd, id);
              if (!dest) return;
              xs.push(origin.x, dest.x, null);
              ys.push(origin.y, dest.y, null);
            });
            const visible = xs.length > 0;
            Plotly.restyle(gd, { x: [xs], y: [ys], visible: [visible] }, [overlay.lineIdx]);
          });
        }

        function applyForId(globalId) {
          if (!toggle.checked) {
            applyGlobalIdHighlight(gd, null);
            hideOverlay();
            return;
          }
          if (!Number.isFinite(globalId)) {
            applyGlobalIdHighlight(gd, null);
            hideOverlay();
            return;
          }
          const key = String(globalId);
          const activeProfile = gd.__substitutionProfile && neighborsByProfile[gd.__substitutionProfile]
            ? gd.__substitutionProfile
            : resolveDefaultSubstitutionProfile(neighborsByProfile, profileConfig);
          const activeMap = activeProfile ? neighborsByProfile[activeProfile] : null;
          const entries = activeMap ? (activeMap[key] || []) : [];
          if (!entries.length) {
            const ownSet = new Set([Number(globalId)]);
            applyGlobalIdHighlight(gd, ownSet, 0.1);
            hideOverlay();
            return;
          }
          const active = new Set([Number(globalId)]);
           const neighborIds = [];
          entries.forEach(item => {
            if (item && Object.prototype.hasOwnProperty.call(item, 'neighbor')) {
              const neigh = Number(item.neighbor);
              active.add(neigh);
              neighborIds.push(neigh);
            }
          });
          applyGlobalIdHighlight(gd, active, 0.1);
          drawLines(globalId, neighborIds);
        }

        gd.on('plotly_hover', ev => {
          const pt = ev.points && ev.points[0];
          if (!pt || !pt.customdata || pt.customdata.length < 8) {
            lastHoverId = null;
            if (toggle.checked) {
              applyGlobalIdHighlight(gd, null);
              hideOverlay();
            }
            return;
          }
          const gid = Number(pt.customdata[7]);
          lastHoverId = gid;
          if (toggle.checked) {
            applyForId(gid);
          }
        });

        gd.on('plotly_unhover', () => {
          lastHoverId = null;
          if (toggle.checked) {
            applyGlobalIdHighlight(gd, null);
            hideOverlay();
          }
        });

        toggle.addEventListener('change', () => {
          if (!toggle.checked) {
            applyGlobalIdHighlight(gd, null);
            hideOverlay();
            return;
          }
          if (Number.isFinite(lastHoverId)) {
            applyForId(lastHoverId);
          }
        });
      }
      function registerCardHighlight(card) {
        const figures = card.querySelectorAll('.js-plotly-plot');
        figures.forEach(gd => {
            const attach = () => {
                if (card.dataset.familyHighlight === '1') {
                    const info = gd.layout && gd.layout.meta && gd.layout.meta.familyHighlight;
                    if (info && info.enabled) {
                        setupFamilyHighlight(gd);
                    }
                }
                setupInversionHighlight(gd);
                setupSubstitutionHighlight(gd);
            };
            if (gd.layout && gd.layout.meta) {
                attach();
            } else {
                const handler = () => {
                    gd.removeListener('plotly_afterplot', handler);
                    attach();
                };
                gd.on('plotly_afterplot', handler);
            }
        });
      }

      function registerCardDetail(card) {
        const detailPanel = card.querySelector('.detail-panel');
        if (!detailPanel) return;
        const defaultMsg = detailPanel.dataset.defaultMsg || 'Haz clic en un punto para ver el detalle completo.';
        detailPanel.innerHTML = defaultMsg;
        const figures = card.querySelectorAll('.js-plotly-plot');
        figures.forEach(gd => {
          const neighborsByProfile = (gd.layout && gd.layout.meta && gd.layout.meta.substitutionNeighbors) || null;
          const profileConfig = (gd.layout && gd.layout.meta && gd.layout.meta.substitutionProfiles) || {};
          let lastDetailState = null;

          function lookupLabelById(id) {
            // Busca en las trazas visibles un punto con ese global_id y devuelve su 'text'
            for (let i = 0; i < gd.data.length; i++) {
              const trace = gd.data[i];
              const cd = trace.customdata || [];
              const tx = trace.text || [];
              for (let j = 0; j < cd.length; j++) {
                const row = cd[j];
                if (row && row.length >= 8 && row[7] === id) {
                  return typeof tx[j] === 'string' ? tx[j] : `Acorde ID: ${id}`;
                }
              }
            }
            return `Acorde ID: ${id}`;
          }

          function getActiveProfileKey() {
            if (neighborsByProfile) {
              if (gd.__substitutionProfile && neighborsByProfile[gd.__substitutionProfile]) {
                return gd.__substitutionProfile;
              }
              if (profileConfig && profileConfig.default && neighborsByProfile[profileConfig.default]) {
                return profileConfig.default;
              }
              const keys = Object.keys(neighborsByProfile);
              if (keys.length) return keys[0];
            }
            return null;
          }

          function getProfileLabel(profileKey) {
            if (!profileKey || !profileConfig || !profileConfig.profiles) return profileKey;
            const info = profileConfig.profiles[profileKey];
            return info && info.label ? info.label : profileKey;
          }

          function getSubstitutionList(globalId) {
            if (!Number.isFinite(globalId) || !neighborsByProfile) return [];
            const profileKey = getActiveProfileKey();
            if (!profileKey) return [];
            const profileMap = neighborsByProfile[profileKey] || {};
            return profileMap[String(globalId)] || [];
          }

          const updatePanel = (content, musicalInversions, structuralInversions, currentGlobalId) => {
            let html = content || defaultMsg;
            if (musicalInversions && musicalInversions.length > 0) {
                html += "<h5>Inversiones Musicales</h5><ul>";
                musicalInversions.forEach(id => { html += `<li>${lookupLabelById(id)}</li>`; });
                html += "</ul>";
            }
            if (structuralInversions && structuralInversions.length > 0) {
                html += "<h5>Inversiones Estructurales</h5><ul>";
                structuralInversions.forEach(id => { html += `<li>${lookupLabelById(id)}</li>`; });
                html += "</ul>";
            }
            if (Number.isFinite(currentGlobalId)) {
                const subList = getSubstitutionList(currentGlobalId);
                if (subList.length > 0) {
                    const activeKey = getActiveProfileKey();
                    const profileLabel = activeKey ? getProfileLabel(activeKey) : null;
                    const titleSuffix = profileLabel ? ` (${profileLabel})` : "";
                    html += `<h5>Sustitutos sugeridos${titleSuffix}</h5><ol>`;
                    subList.forEach(item => {
                        const label = lookupLabelById(Number(item.neighbor));
                        const dist = typeof item.distance === 'number' ? item.distance.toFixed(3) : '—';
                        html += `<li>${label} (dist: ${dist})</li>`;
                    });
                    html += "</ol>";
                }
            }
            detailPanel.innerHTML = html;
            lastDetailState = {
              content: content || defaultMsg,
              musicalInversions: musicalInversions || [],
              structuralInversions: structuralInversions || [],
              currentGlobalId,
            };
          };
          gd.on('plotly_click', ev => {
            const pt = ev.points && ev.points[0];
            if (!pt || !pt.customdata || pt.customdata.length < 7) {
              updatePanel(defaultMsg, null, null, null);
              return;
            }
            const musicalInversions = pt.customdata[5];
            const structuralInversions = pt.customdata[6];
            const currentId = pt.customdata[7];
            updatePanel(pt.customdata[4], musicalInversions, structuralInversions, currentId);
          });
          gd.on('plotly_doubleclick', () => updatePanel(defaultMsg, null, null, null));
          card.addEventListener('substitutionProfileChanged', () => {
            if (lastDetailState && Number.isFinite(lastDetailState.currentGlobalId)) {
              updatePanel(
                lastDetailState.content,
                lastDetailState.musicalInversions,
                lastDetailState.structuralInversions,
                lastDetailState.currentGlobalId,
              );
            }
          });
        });
      }

      // --------------------------------------------------------------------------------
      // Dynamic Heatmap Coordination (lazy load + visibility aware)
      // --------------------------------------------------------------------------------
      const heatmapScriptCache = {};
      function loadHeatmapPayload(scenarioName, path) {
        if (!path) return Promise.resolve(null);
        if (window.__HEATMAP_PAYLOADS && window.__HEATMAP_PAYLOADS[scenarioName]) {
          return Promise.resolve(window.__HEATMAP_PAYLOADS[scenarioName]);
        }
        if (!heatmapScriptCache[scenarioName]) {
          heatmapScriptCache[scenarioName] = new Promise(resolve => {
            const script = document.createElement('script');
            script.src = path;
            script.onload = () => {
              const payload = window.__HEATMAP_PAYLOADS ? window.__HEATMAP_PAYLOADS[scenarioName] : null;
              resolve(payload || null);
            };
            script.onerror = () => resolve(null);
            document.head.appendChild(script);
          });
        }
        return heatmapScriptCache[scenarioName];
      }

      function registerDynamicHeatmap(card) {
        const rootData = window.HEATMAP_DATA || {};
        const meta = rootData.metadata || {};
        const files = rootData.files || {};
        const labels = Array.isArray(meta.labels) ? meta.labels : [];
        if (!labels.length) return;

        const header = card.querySelector('.card-header strong');
        if (!header) return;
        const scenarioName = header.textContent.trim();
        const filePath = files[scenarioName];
        if (!filePath) return;

        const heatmapContainer = card.querySelector('.aux-figure[data-section="heatmap"] .js-plotly-plot');
        if (!heatmapContainer) return;

        const scatters = Array.from(card.querySelectorAll('.subtab-panel .js-plotly-plot'));
        if (!scatters.length) return;

        const cardinalities = Array.isArray(meta.cardinalities) ? meta.cardinalities : [];
        const totalPoints = labels.length;

        function getBaseDimension(axis, fallback) {
          const layout = heatmapContainer._fullLayout || heatmapContainer.layout || {};
          if (typeof layout[axis] === 'number' && layout[axis] > 0) {
            return layout[axis];
          }
          if (heatmapContainer.__heatmapDims && typeof heatmapContainer.__heatmapDims[axis] === 'number') {
            return heatmapContainer.__heatmapDims[axis];
          }
          const measured = axis === 'width' ? heatmapContainer.clientWidth : heatmapContainer.clientHeight;
          if (measured && measured > 0) {
            heatmapContainer.__heatmapDims = heatmapContainer.__heatmapDims || {};
            heatmapContainer.__heatmapDims[axis] = measured;
            return measured;
          }
          return fallback;
        }

        function isPlotVisible(gd) {
          if (!gd) return false;
          let node = gd;
          while (node && node !== card) {
            if (node.nodeType !== 1) {
              node = node.parentElement;
              continue;
            }
            const style = window.getComputedStyle(node);
            const opacity = parseFloat(style.opacity || '1');
            if (style.display === 'none' || style.visibility === 'hidden' || opacity === 0) {
              return false;
            }
            node = node.parentElement;
          }
          return true;
        }

        loadHeatmapPayload(scenarioName, filePath).then(payload => {
          if (!payload || !Array.isArray(payload.condensed) || !payload.condensed.length) {
            return;
          }
          if (card.__heatmapBound) return;
          card.__heatmapBound = true;

          const condensed = payload.condensed;
          const matrixSize = Math.round((1 + Math.sqrt(1 + 8 * condensed.length)) / 2);
          if (!Number.isFinite(matrixSize) || matrixSize <= 0) {
            return;
          }

          function getDistance(idx1, idx2) {
            if (!Number.isFinite(idx1) || !Number.isFinite(idx2)) return NaN;
            if (idx1 === idx2) return 0;
            let i = Math.min(idx1, idx2);
            let j = Math.max(idx1, idx2);
            const k = Math.round((matrixSize * i) - (i * (i + 1) / 2) + j - i - 1);
            if (k < 0 || k >= condensed.length) {
              return NaN;
            }
            return condensed[k];
          }

          function getCardinality(idx) {
            if (cardinalities && Number.isFinite(cardinalities[idx])) {
              return Number(cardinalities[idx]);
            }
            const label = labels[idx] || '';
            const match = label.match(/\[(.*?)\]/);
            if (match) {
              const parts = match[1].split(',').filter(Boolean);
              return parts.length + 1;
            }
            const first = label.split(' ')[0] || '';
            return first.length || 0;
          }

          function dedupeValidIndices(source) {
            const unique = new Set();
            source.forEach(idx => {
              if (Number.isFinite(idx) && idx >= 0 && idx < totalPoints) {
                unique.add(Number(idx));
              }
            });
            return Array.from(unique);
          }

          function extractIndexFromTrace(trace, pointIdx) {
            if (!trace) return null;
            if (trace.customdata && trace.customdata[pointIdx] && trace.customdata[pointIdx].length >= 8) {
              const value = Number(trace.customdata[pointIdx][7]);
              if (Number.isFinite(value)) {
                return value;
              }
            }
            if (typeof pointIdx === 'number') {
              return pointIdx;
            }
            return null;
          }

          function collectIndicesFromPoints(gd, points) {
            const indices = [];
            if (!Array.isArray(points)) {
              return indices;
            }
            points.forEach(pt => {
              if (!pt) return;
              if (pt.customdata && pt.customdata.length >= 8) {
                const candidate = Number(pt.customdata[7]);
                if (Number.isFinite(candidate)) {
                  indices.push(candidate);
                  return;
                }
              }
              if (gd && gd.data && typeof pt.curveNumber === 'number') {
                const trace = gd.data[pt.curveNumber];
                const fallback = extractIndexFromTrace(trace, pt.pointIndex);
                if (Number.isFinite(fallback)) {
                  indices.push(fallback);
                }
              }
            });
            return indices;
          }

          function collectIndicesInRange(gd, xRange, yRange) {
            const indices = [];
            const hasRange =
              Array.isArray(xRange) &&
              xRange.length === 2 &&
              Array.isArray(yRange) &&
              yRange.length === 2;
            if (!gd || !Array.isArray(gd.data) || !hasRange) {
              return indices;
            }
            const xMin = Math.min(Number(xRange[0]), Number(xRange[1]));
            const xMax = Math.max(Number(xRange[0]), Number(xRange[1]));
            const yMin = Math.min(Number(yRange[0]), Number(yRange[1]));
            const yMax = Math.max(Number(yRange[0]), Number(yRange[1]));
            if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) {
              return indices;
            }
            gd.data.forEach(trace => {
              if (!trace || !Array.isArray(trace.x) || !Array.isArray(trace.y)) {
                return;
              }
              for (let i = 0; i < trace.x.length; i++) {
                const xVal = Number(trace.x[i]);
                const yVal = Number(trace.y[i]);
                if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) {
                  continue;
                }
                if (xVal >= xMin && xVal <= xMax && yVal >= yMin && yVal <= yMax) {
                  const idx = extractIndexFromTrace(trace, i);
                  if (Number.isFinite(idx)) {
                    indices.push(idx);
                  }
                }
              }
            });
            return indices;
          }

          function getVisibleIndexSet() {
            const visible = new Set();
            let hasAny = false;
            scatters.forEach(gd => {
              if (!gd || !Array.isArray(gd.data)) {
                return;
              }
              if (!isPlotVisible(gd)) {
                return;
              }
              gd.data.forEach(trace => {
                if (!trace || trace.visible === 'legendonly' || trace.visible === false) {
                  return;
                }
                const len = Array.isArray(trace.x) ? trace.x.length : 0;
                for (let i = 0; i < len; i++) {
                  const idx = extractIndexFromTrace(trace, i);
                  if (Number.isFinite(idx)) {
                    visible.add(idx);
                    hasAny = true;
                  }
                }
              });
            });
            return hasAny ? visible : null;
          }

          let lastSubset = null;

          function requestUpdate(indices) {
            if (Array.isArray(indices)) {
              lastSubset = indices.slice();
            } else {
              lastSubset = null;
            }
            updateHeatmap(indices);
          }

          function updateHeatmap(subsetIndicesRaw) {
            let subsetIndices = dedupeValidIndices(subsetIndicesRaw || []);
            const visibleSet = getVisibleIndexSet();

            if (!subsetIndices.length) {
              if (visibleSet && visibleSet.size) {
                subsetIndices = Array.from(visibleSet);
              } else {
                subsetIndices = dedupeValidIndices(labels.map((_, i) => i));
              }
            } else if (visibleSet && visibleSet.size) {
              subsetIndices = subsetIndices.filter(i => visibleSet.has(i));
              if (!subsetIndices.length) {
                subsetIndices = Array.from(visibleSet);
              }
            }

            if (!subsetIndices.length) {
              subsetIndices = [0];
            }

            let items = subsetIndices.map(i => ({
              originalIndex: i,
              label: labels[i],
              n: getCardinality(i),
            }));

            items.sort((a, b) => a.n - b.n);
            const size = items.length;
            const axesLabels = items.map(x => x.label);
            const z = new Array(size);
            const text = new Array(size);

            for (let r = 0; r < size; r++) {
              z[r] = new Array(size);
              text[r] = new Array(size);
              for (let c = 0; c < size; c++) {
                if (c < r) {
                  z[r][c] = NaN;
                  text[r][c] = '';
                  continue;
                }
                const dist = getDistance(items[r].originalIndex, items[c].originalIndex);
                z[r][c] = Number.isFinite(dist) ? dist : 0;
                text[r][c] = `${axesLabels[r]}<br>vs ${axesLabels[c]}`;
              }
            }

            const showLabels = size <= 30;
            const tickVals = showLabels ? items.map((_, i) => i) : null;
            const tickText = showLabels ? axesLabels : null;
            const layoutWidth = getBaseDimension('width', 640);
            const layoutHeight = getBaseDimension('height', 480);

            Plotly.react(heatmapContainer, [{
              type: 'heatmap',
              z: z,
              x: items.map((_, i) => i),
              y: items.map((_, i) => i),
              colorscale: 'Turbo',
              colorbar: { title: 'Distancia' },
              hoverongaps: false,
              text: text,
              hovertemplate: "%{text}<br>Distancia=%{z:.3f}<extra></extra>",
            }], {
              title: `Heatmap: ${scenarioName} (Subset N=${size})`,
              template: 'plotly_white',
              width: layoutWidth,
              height: layoutHeight,
              xaxis: {
                tickmode: 'array',
                tickvals: tickVals,
                ticktext: tickText,
                showticklabels: showLabels,
              },
              yaxis: {
                tickmode: 'array',
                tickvals: tickVals,
                ticktext: tickText,
                showticklabels: showLabels,
                autorange: 'reversed',
              },
            });
          }

          scatters.forEach(gd => {
            gd.on('plotly_selected', (eventData) => {
              if (!eventData || !eventData.points || eventData.points.length === 0) {
                requestUpdate([]);
                return;
              }
              const indices = collectIndicesFromPoints(gd, eventData.points);
              requestUpdate(indices);
            });

            gd.on('plotly_deselect', () => {
              requestUpdate([]);
            });

            gd.on('plotly_relayout', (eventData = {}) => {
              const hasXBracket =
                Object.prototype.hasOwnProperty.call(eventData, 'xaxis.range[0]') &&
                Object.prototype.hasOwnProperty.call(eventData, 'xaxis.range[1]');
              const hasYBracket =
                Object.prototype.hasOwnProperty.call(eventData, 'yaxis.range[0]') &&
                Object.prototype.hasOwnProperty.call(eventData, 'yaxis.range[1]');
              const hasXArray = Array.isArray(eventData['xaxis.range']) && eventData['xaxis.range'].length === 2;
              const hasYArray = Array.isArray(eventData['yaxis.range']) && eventData['yaxis.range'].length === 2;
              if ((hasXBracket || hasXArray) && (hasYBracket || hasYArray)) {
                const xRange = hasXBracket
                  ? [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']]
                  : eventData['xaxis.range'];
                const yRange = hasYBracket
                  ? [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']]
                  : eventData['yaxis.range'];
                const indices = collectIndicesInRange(gd, xRange, yRange);
                requestUpdate(indices);
                return;
              }
              if (eventData['xaxis.autorange'] === true || eventData['yaxis.autorange'] === true) {
                requestUpdate([]);
              }
            });

            gd.on('plotly_restyle', () => {
              if (lastSubset === null) {
                requestUpdate(null);
              } else {
                requestUpdate(lastSubset.slice());
              }
            });

            gd.on('plotly_legendclick', () => {
              setTimeout(() => {
                if (lastSubset === null) {
                  requestUpdate(null);
                } else {
                  requestUpdate(lastSubset.slice());
                }
              }, 0);
            });

            gd.on('plotly_legenddoubleclick', () => {
              setTimeout(() => {
                requestUpdate(null);
              }, 0);
            });
          });

          requestUpdate(null);
        });
      }

      document.querySelectorAll('.plot-card').forEach(card => {
        registerCardFilters(card);
        registerCardHighlight(card);
        registerCardDetail(card);
        registerDynamicHeatmap(card); // New function
      });
    })();
