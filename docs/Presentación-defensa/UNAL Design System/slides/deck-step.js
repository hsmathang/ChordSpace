/**
 * useDeckStep(maxStep, slideId)
 *
 * Shared navigation hook for animated slides inside <deck-stage>.
 *
 * Contract:
 * - The deck owns slide-to-slide navigation.
 * - The active animated slide owns its internal reveal steps.
 * - Hidden slides never receive keyboard steps.
 * - A slide resets to step 0 every time it becomes active.
 * - When a slide becomes active from a held/repeated key press, it waits for
 *   keyup before accepting reveal steps. This prevents the next slide from
 *   arriving already fully revealed.
 *
 * Usage in a slide component:
 *
 *   const MySlide = () => {
 *     const step = window.useDeckStep(4, 'slide-my-slide');
 *     return <div>{step >= 1 && <Something />}</div>;
 *   };
 */
window.useDeckStep = function useDeckStep(maxStep, slideId) {
  const [step, setStep] = React.useState(0);
  const [isActive, setIsActive] = React.useState(false);
  const isActiveRef = React.useRef(false);
  const waitForKeyReleaseRef = React.useRef(false);

  React.useEffect(() => {
    const slide = document.getElementById(slideId);
    const syncActive = (event) => {
      const active = !!slide?.hasAttribute('data-deck-active');
      const wasActive = isActiveRef.current;
      const reason = event?.detail?.reason || 'init';

      isActiveRef.current = active;
      setIsActive(active);

      if (active && !wasActive) {
        setStep(0);
        waitForKeyReleaseRef.current = reason !== 'init';
      }
    };

    syncActive();
    const syncTimer = window.setTimeout(syncActive, 80);
    const deck = document.querySelector('deck-stage');
    deck?.addEventListener('slidechange', syncActive);
    return () => {
      window.clearTimeout(syncTimer);
      deck?.removeEventListener('slidechange', syncActive);
    };
  }, [slideId]);

  React.useEffect(() => {
    const isDeckKey = (e) => (
      e.key === 'ArrowRight' ||
      e.key === 'ArrowLeft' ||
      e.key === ' ' ||
      e.key === 'PageDown' ||
      e.key === 'PageUp'
    );

    const handleKey = (e) => {
      if (!isActiveRef.current) return;
      if (e.target && (e.target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName))) return;

      if (waitForKeyReleaseRef.current && isDeckKey(e)) {
        e.stopPropagation();
        e.preventDefault();
        return;
      }

      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
        if (step < maxStep) {
          e.stopPropagation();
          e.preventDefault();
          setStep((s) => Math.min(maxStep, s + 1));
        }
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        if (step > 0) {
          e.stopPropagation();
          e.preventDefault();
          setStep((s) => Math.max(0, s - 1));
        }
      }
    };

    const handleKeyUp = (e) => {
      if (!isActiveRef.current) return;
      if (isDeckKey(e)) waitForKeyReleaseRef.current = false;
    };

    window.addEventListener('keydown', handleKey, { capture: true });
    window.addEventListener('keyup', handleKeyUp, { capture: true });
    return () => {
      window.removeEventListener('keydown', handleKey, { capture: true });
      window.removeEventListener('keyup', handleKeyUp, { capture: true });
    };
  }, [isActive, maxStep, step]);

  return step;
};
