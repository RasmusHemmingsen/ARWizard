using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public enum GestureType
{
    Fireball,
    Frostball,
    Shoot,
    Grassball
}

public class Gesture
{
    public float Percentage { get; set; }
    public GestureType Type { get; set; }

    public Gesture() { }

    public Gesture(string name, float percentage)
    {
        Percentage = Mathf.Clamp(percentage, 0, 100);
        Type = (GestureType)int.Parse(name);
    }
}

public class Spell
{
    public GameObject channelPrefab { get; set; }
    public GameObject spellPrefab { get; set; }

    public Spell(GameObject channel, GameObject spell)
    {
        channelPrefab = channel;
        spellPrefab = spell;
    }
}

public class MainGameManager : MonoBehaviour
{
    [Range(0f, 1f)]
    public float percentage = 0.8f;

    [SerializeField]
    private string targetNameInHieracy;

    [SerializeField]
    int concurrentParticles = 20;

    [SerializeField]
    GameObject channelFireballParticlePrefab, fireballPrefab, channelFrostballParticlePrefab, frostballPrefab, channelGrassParticlePrefab, grassballPrefab;

    private GameObject targetHand;
    private bool _isSpellActive;
    private Spell _currentSpell;
    private List<GameObject> activeParticles;

    private Queue<Vector3> previousPositions;

    private void Awake()
    {
        previousPositions = new Queue<Vector3>();
        activeParticles = new List<GameObject>();
        HandThroughLeap.HandGesturePercentageEvent += OnEvent;
        _isSpellActive = false;
    }

    private void Start()
    {
        StartCoroutine(SavePreviousHandPos());
        targetHand = GameObject.Find(targetNameInHieracy);
    }

    IEnumerator SavePreviousHandPos()
    {
        while (true)
        {
            while (targetHand != null)
            {
                previousPositions.Enqueue(targetHand.transform.position);
                if (previousPositions.Count > 10)
                {
                    previousPositions.Dequeue();
                }
                yield return new WaitForSeconds(0.05f);
            }
            yield return new WaitForSeconds(0.1f);
        }
    }

    private void Update()
    {
        if (targetHand == null)
        {
            targetHand = GameObject.Find(targetNameInHieracy);
        }
    }

    private void OnEvent(Gesture g)
    {
        if (g.Percentage < percentage)
            return;
        if (_isSpellActive && g.Type == GestureType.Shoot)
        {
            Shoot();
            return;
        }
        if(_isSpellActive)
            return;

        _isSpellActive = true;
        _currentSpell = GetSpellPrefab(g.Type);
        ChannelSpell(_currentSpell);
    }

    private void Shoot()
    {
        foreach (var particle in activeParticles)
        {
            Destroy(particle);
        }
        activeParticles = new List<GameObject>();

        var spellPrefab = Instantiate(_currentSpell.spellPrefab);
        spellPrefab.transform.position = targetHand.transform.position;
        //BallSpell.GetComponent<BallSpell>().Shoot((targetHand.transform.position - previousPositions.Peek()));
        spellPrefab.GetComponent<BallSpell>().Shoot(-targetHand.transform.up);
        _isSpellActive = false;
    }

    private void ChannelSpell(Spell spell)
    {
        for (var i = 0; i < concurrentParticles; i++)
        {
            var particle = Instantiate(spell.channelPrefab);
            activeParticles.Add(particle);
            particle.transform.position = targetHand.transform.position + new Vector3(Random.Range(-0.05f, 0.05f), Random.Range(-0.05f, 0.05f), Random.Range(-0.05f, 0.05f));
        }
    }

    private Spell GetSpellPrefab(GestureType gestureType)
    {
        Spell spell;
        switch (gestureType)
        {
            case GestureType.Fireball:
                spell = new Spell(channelFireballParticlePrefab, fireballPrefab);

                break;
            case GestureType.Frostball:
                spell = new Spell(channelFrostballParticlePrefab, frostballPrefab);
                break;
            case GestureType.Grassball:
                spell = new Spell(channelGrassParticlePrefab, grassballPrefab);
                break;
            default:
                return null;
        }

        return spell;
    }
}