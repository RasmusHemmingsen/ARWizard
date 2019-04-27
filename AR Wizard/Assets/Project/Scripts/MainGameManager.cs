using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public enum GestureType
{
    Fireball,
    Frostball,
    Shoot,
    Nothing
}

public class Gesture
{
    private float _percentage;
    public float Percentage
    {
        get => _percentage;
        set => _percentage = value * 100;
    }

    public GestureType Type { get; set; }

    public Gesture() { }

    public Gesture(string name, float percentage)
    {
        Percentage = Mathf.Clamp(percentage, 0, 100);
        Type = (GestureType)int.Parse(name);
    }
}

public class MainGameManager : MonoBehaviour
{
    [Range(0f, 100f)]
    public float percentage = 80f;

    [SerializeField]
    private string targetNameInHieracy;

    [SerializeField]
    int concurrentParticles = 20;

    [SerializeField]
    GameObject channelFireballParticlePrefab, fireballPrefab;

    private GameObject targetHand;
    private GestureType activeSpell;
    private List<GameObject> activeParticles;

    private Queue<Vector3> previousPositions;

    private void Awake()
    {
        previousPositions = new Queue<Vector3>();
        activeParticles = new List<GameObject>();
        HandThroughLeap.HandGesturePercentageEvent += OnEvent;
        activeSpell = GestureType.Nothing;
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
        switch (g.Type)
        {
            case GestureType.Fireball:
                {
                    if (activeSpell != GestureType.Fireball)
                    {
                        activeSpell = GestureType.Fireball;
                        ChannelFireball();
                    }
                    break;
                }
            case GestureType.Frostball:
                {
                    break;
                }
            case GestureType.Shoot:
                {
                    if (activeSpell == GestureType.Fireball)
                    {
                        ShootFireball();
                    }
                    break;
                }
            default:
                break;
        }
    }

    private void ChannelFireball()
    {
        for (int i = 0; i < concurrentParticles; i++)
        {
            var particle = Instantiate(channelFireballParticlePrefab);
            activeParticles.Add(particle);
            particle.transform.position = targetHand.transform.position + new Vector3(Random.Range(-0.05f, 0.05f), Random.Range(-0.05f, 0.05f), Random.Range(-0.05f, 0.05f));
        }
    }

    private void ShootFireball()
    {
        foreach (var particle in activeParticles)
        {
            Destroy(particle);
        }
        activeParticles = new List<GameObject>();

        var fireball = Instantiate(fireballPrefab);
        fireball.transform.position = targetHand.transform.position;
        fireball.GetComponent<Fireball>().Shoot((targetHand.transform.position - previousPositions.Peek()));
        activeSpell = GestureType.Nothing;
    }
}